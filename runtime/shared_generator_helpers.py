"""Compose first-use dependencies for a shared generator helper and a dialect callback."""

import ast
import copy
import marshal
import tempfile
from pathlib import Path

import dispatch_helpers
import generator_cells
import generator_contexts
import import_cone
import reflective_use
import structural_dependencies
import symbol_impact

SHARED_FILE = "sqlglot/dialects/dialect.py"


def generator_class(node):
    return (
        isinstance(node, ast.ClassDef)
        and node.name.endswith("Generator")
        and not node.decorator_list
        and not node.keywords
    )


def transform_assignment(node):
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "TRANSFORMS"
    )


def eager_table_read(statements):
    for stmt in statements:
        early = (
            [stmt.args, *stmt.decorator_list]
            if isinstance(stmt, ast.FunctionDef)
            else [stmt]
        )
        if any(
            isinstance(n, (ast.Name, ast.Attribute))
            and getattr(n, "id", getattr(n, "attr", "")) == "TRANSFORMS"
            for expr in early
            for n in ast.walk(expr)
        ):
            return True
    return False


def bindings(sel, name):
    """Require every static reference to be a verified import or a named dispatch cell."""
    grouped = {}
    for path, line, scope, kind in symbol_impact.index().get(name, ()):
        if not path.startswith("sqlglot/generators/") or scope:
            return None
        grouped.setdefault(path, []).append((line, kind))
    slots = []
    for path, refs in grouped.items():
        _, tree = sel.source(path)
        found = set()
        handler_nodes = set()
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "sqlglot.dialects.dialect"
                and not stmt.level
            ):
                if any(a.name == name and a.asname is None for a in stmt.names):
                    found.add((stmt.lineno, "from:" + stmt.module))
            if not generator_class(stmt):
                continue
            for idx, assign in enumerate(stmt.body):
                if not transform_assignment(assign) or not isinstance(
                    assign.value, ast.Dict
                ):
                    continue
                for key, handler in zip(assign.value.keys, assign.value.values):
                    if (
                        isinstance(handler, ast.Name)
                        and handler.id == name
                        and isinstance(key, ast.Attribute)
                        and isinstance(key.value, ast.Name)
                        and key.value.id == "exp"
                    ):
                        if eager_table_read(stmt.body[idx + 1 :]):
                            return None
                        handler_nodes.add(handler)
                        found.add((handler.lineno, "name"))
                        slots.append((path, stmt.name, key.attr))
        all_loads = {
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
        }
        if all_loads != handler_nodes or set(refs) != found:
            return None
        if any(
            isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name == name
            or isinstance(n, ast.Name)
            and n.id == name
            and isinstance(n.ctx, (ast.Del, ast.Store))
            or isinstance(n, ast.alias)
            and (n.asname == name or n.name == "*")
            for n in ast.walk(tree)
        ):
            return None
    return slots or None


def projections(sel, change, apply, equal):
    old, before = sel.source(SHARED_FILE)
    text = apply(old, change)[0]
    if text != change.get("_compiled_source"):
        return None
    after = ast.parse(text)
    if len(before.body) != len(after.body):
        return None
    slots = []
    body_readers = set()
    for left, right in zip(before.body, after.body):
        if equal(left, right):
            continue
        if not (
            isinstance(left, ast.FunctionDef)
            and isinstance(right, ast.FunctionDef)
            and generator_contexts.header(left, right, equal)
            and left.name.endswith("_sql")
            and isinstance(left.args.args[0].annotation, ast.Name)
            and left.args.args[0].annotation.id == "Generator"
            and ast.get_docstring(left, clean=False)
            == ast.get_docstring(right, clean=False)
        ):
            return None
        if not all(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "unsupported_args"
            and not d.keywords
            and all(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                for arg in d.args
            )
            for d in left.decorator_list
        ):
            return None
        for fn in (left, right):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr
                in {
                    "set",
                    "pop",
                    "append",
                    "remove",
                    "update",
                    "extend",
                    "clear",
                    "__setitem__",
                }
                for n in ast.walk(fn)
            ):
                return None
            plain = copy.copy(fn)
            plain.decorator_list = []
            if not import_cone.local_function(plain):
                return None
        body_lines = range(left.body[0].lineno, left.end_lineno + 1)
        if not set(body_lines).isdisjoint(
            sel.mapping.get("import_lines", {}).get(SHARED_FILE, ())
        ):
            return None
        body_readers.update(sel.line_tests(SHARED_FILE, body_lines))
        cells = bindings(sel, left.name)
        if not cells:
            return None
        slots.extend(cells)
    return (slots, body_readers) if slots else None


def frame(node, keys):
    """Mask only the selected callback values in a fixed, passive dictionary comprehension."""
    copied = copy.deepcopy(node)
    literal = dispatch_helpers.literal_mapping(copied)
    literal.values = [
        ast.Constant(None) if isinstance(k, ast.Attribute) and k.attr in keys else value
        for k, value in zip(literal.keys, literal.values)
    ]
    return copied


def table_changes(sel, change, apply, equal):
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/generators/")
        or not path.endswith(".py")
    ):
        return None
    raw, old = sel.source(path)
    text = apply(raw, change)[0]
    if change.get("_compiled_source") != text:
        return None
    new = ast.parse(text)
    if len(old.body) != len(new.body):
        return None
    found = []
    for left, right in zip(old.body, new.body):
        if equal(left, right):
            continue
        if not (
            generator_class(left)
            and generator_class(right)
            and generator_contexts.header(left, right, equal)
            and len(left.body) == len(right.body)
        ):
            return None
        for idx, (a, b) in enumerate(zip(left.body, right.body)):
            if equal(a, b):
                continue
            if not (
                transform_assignment(a)
                and transform_assignment(b)
                and equal(a.targets, b.targets)
            ):
                return None
            if eager_table_read(left.body[idx + 1 :]):
                return None
            x = dispatch_helpers.literal_mapping(a.value)
            y = dispatch_helpers.literal_mapping(b.value)
            if x is None or y is None:
                return None
            p, q = copy.copy(a), copy.copy(b)
            p.value = x
            q.value = y
            keys = generator_cells.differences(
                p, q, equal, generator_cells.factory_bindings(old)
            )
            if not keys or not equal(frame(a.value, keys), frame(b.value, keys)):
                return None
            found.extend((path, left.name, k) for k in keys)
    return found or None


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    if len(changes) != 2 or remaining < 4 or not selected:
        return selected
    shared = next((c for c in changes if c["old"] == c["new"] == SHARED_FILE), None)
    if shared is None:
        return selected
    other = next(c for c in changes if c is not shared)
    helpers = projections(sel, shared, apply, equal)
    if not helpers:
        return selected
    edits = table_changes(sel, other, apply, equal)
    if not edits:
        return selected
    helper_slots, body_readers = helpers
    slots = helper_slots + edits
    profile = generator_contexts.data_dispatch()
    keys = {key for _, _, key in slots}
    if profile["commit"] != sel.mapping["commit"] or keys & set(
        profile["structural"]["dialect_init_keys"]
    ):
        return selected
    if any(
        not profile["sources"].get(p)
        or profile["sources"][p] != sel.mapping["source_hashes"].get(p)
        for p, _, _ in slots
    ):
        return selected
    custom = import_cone.receiver_users(sel, [])
    if custom is None:
        return selected
    keep = set(sel.confirmed_failures) | sel.unknown | custom | body_readers
    reflect = generator_contexts.data("parser-cells.json")
    keep.update(
        sel.by_name[reflect["names"][i]]
        for i in reflective_use.indices(sel, reflect)
        if reflect["names"][i] in sel.by_name
    )
    paths = {SHARED_FILE} | {p for p, _, _ in slots}
    ext = sel.dispatch_map()
    keep.update(ext.external)
    for path in paths:
        sel.load_shard(path)
        if path in sel.files:
            keep.update(sel.sets[sel.files[path]])
        keep.update(ext.external_by_source.get(path, ()))
    for path, owner, key in slots:
        keep.update(
            structural_dependencies.table_observers(
                sel, profile, path, path[:-3].replace("/", ".") + ":" + owner, {key}
            )
        )
    if selected <= keep:
        return selected
    old = generator_cells.baseline()
    with tempfile.TemporaryDirectory(prefix="generator-shared-") as tmp:
        root = Path(tmp) / "src"
        out = Path(tmp) / "state"
        checkout(root, changes, sel)
        generator_cells.forkprobe.run(
            (root, sorted(keys), out), {}, generator_cells.probe, min(3, remaining)
        )
        if not out.exists():
            return selected
        fresh = marshal.loads(out.read_bytes())
    if fresh.keys() != old.keys():
        return selected
    required = {
        (path[:-3].replace("/", ".") + ":" + owner, k) for path, owner, k in slots
    }
    if any(owner not in fresh or k not in fresh[owner] for owner, k in required):
        return selected
    for owner, values in fresh.items():
        for key, current in values.items():
            prior = old[owner].get(key, [False, "missing", "missing"])
            if current != prior or None in prior or None in current:
                # Nonlocal copied or regenerated registry entries use every original
                # source's cold-reader allowance as well as the effective owner.
                for path in {p for p, _, name in slots if name == key}:
                    keep.update(
                        structural_dependencies.table_observers(
                            sel, profile, path, owner, {key}
                        )
                    )
    proposed = (keep & selected) | sel.confirmed_failures
    if proposed != selected:
        sel.registry_candidates = frozenset(proposed)
    return proposed
