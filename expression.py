"""Expression constants, symbol rendering, syntax highlighting, and expression compilation."""

import ast
import html
import math
import re
from typing import (
    List,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np

from PyQt6.QtGui import QTextCharFormat, QSyntaxHighlighter, QFont, QColor


_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPRESSION_COLUMN_COLOR = "#1d4ed8"
_EXPRESSION_PARAM_COLOR = "#047857"
_EXPRESSION_CONSTANT_COLOR = "#9333ea"
_EXPRESSION_ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "power": np.power,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}
_EXPRESSION_ALLOWED_CONSTANTS = {"pi": float(np.pi), "e": float(np.e)}
_EXPRESSION_HELPER_NAMES = {"col", "columns", "C", "math"}
_LATEX_HTML_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ϵ",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}
_DISPLAY_FUNCTION_NAMES = {
    "abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "sqrt": "sqrt",
    "power": "pow",
    "minimum": "min",
    "maximum": "max",
    "clip": "clip",
}


def _normalize_latex_symbol_token(token_text: str) -> str:
    token = str(token_text).strip()
    if not token:
        return ""
    if token.startswith("$") and token.endswith("$") and len(token) >= 2:
        token = token[1:-1].strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 3:
        token = token[1:-1].strip()
    token = token.strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 3:
        token = token[1:-1].strip()
    return token


def _normalize_latex_script_token(token_text: str) -> str:
    token = str(token_text).strip()
    if not token:
        return ""
    token = re.sub(r"_(?!\{)([A-Za-z0-9+\-]+)", r"_{\1}", token)
    token = re.sub(r"\^(?!\{)([A-Za-z0-9+\-]+)", r"^{\1}", token)
    balance = token.count("{") - token.count("}")
    if balance > 0:
        token = token + ("}" * balance)
    return token


def _strip_outer_braces(text: str) -> str:
    token = str(text).strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 2:
        return token[1:-1].strip()
    return token


def _latex_fragment_to_html(text: str) -> str:
    token = str(text).strip()
    if not token:
        return ""
    if token.startswith("\\"):
        command = token[1:]
        mapped = _LATEX_HTML_COMMANDS.get(command)
        if mapped is not None:
            return mapped
    return html.escape(token)


def latex_symbol_to_plain(symbol_text: str) -> Optional[str]:
    token = _normalize_latex_symbol_token(symbol_text)
    if not token:
        return None
    return _normalize_latex_script_token(token)


def parameter_symbol_to_html(symbol_text: str) -> str:
    token = _normalize_latex_script_token(
        _normalize_latex_symbol_token(str(symbol_text))
    )
    if not token:
        return ""
    match = re.match(
        r"^(?P<base>.+?)(?:_(?P<sub>\{[^{}]*\}|[^_^]+))?(?:\^(?P<sup>\{[^{}]*\}|[^_^]+))?$",
        token,
    )
    if not match:
        return _latex_fragment_to_html(token)
    base = _latex_fragment_to_html(match.group("base") or "")
    sub = _strip_outer_braces(match.group("sub") or "")
    sup = _strip_outer_braces(match.group("sup") or "")
    out = str(base)
    if sub:
        out += f"<sub>{_latex_fragment_to_html(sub)}</sub>"
    if sup:
        out += f"<sup>{_latex_fragment_to_html(sup)}</sup>"
    return out


def parameter_symbol_to_mathtext(symbol_text: str) -> str:
    token = _normalize_latex_script_token(
        _normalize_latex_symbol_token(str(symbol_text))
    )
    if not token:
        return ""
    if token.startswith("$") and token.endswith("$") and len(token) >= 2:
        return token
    return f"${token}$"


def resolve_parameter_symbol(param_key: str, symbol_hint: Optional[str] = None) -> str:
    key_text = str(param_key).strip()
    if symbol_hint is not None:
        raw_symbol = str(symbol_hint).strip()
        if raw_symbol:
            mapped_symbol = latex_symbol_to_plain(raw_symbol)
            return mapped_symbol if mapped_symbol else raw_symbol
    mapped_from_key = latex_symbol_to_plain(key_text)
    return mapped_from_key if mapped_from_key else key_text


def _format_number_literal(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isfinite(value):
            if np.isclose(value, round(value), atol=1e-12):
                return str(int(round(value)))
            return f"{value:.8g}"
        return str(value)
    return str(value)


def _ast_callable_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _ast_callable_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _parenthesize_if_binop(node, rendered: str) -> str:
    if isinstance(node, ast.BinOp):
        return f"({rendered})"
    return rendered


def _parenthesize_if_add_sub(node, rendered: str) -> str:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return f"({rendered})"
    return rendered


def _render_expression_pretty(
    node,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    if isinstance(node, ast.BinOp):
        left = _render_expression_pretty(node.left, name_map=name_map)
        right = _render_expression_pretty(node.right, name_map=name_map)

        if isinstance(node.op, ast.Add):
            return f"{left} + {right}"
        if isinstance(node.op, ast.Sub):
            return f"{left} - {right}"
        if isinstance(node.op, ast.Mult):
            return (
                f"{_parenthesize_if_add_sub(node.left, left)} · "
                f"{_parenthesize_if_add_sub(node.right, right)}"
            )
        if isinstance(node.op, ast.Div):
            return (
                f"{_parenthesize_if_binop(node.left, left)} / "
                f"{_parenthesize_if_binop(node.right, right)}"
            )
        if isinstance(node.op, ast.Pow):
            left_render = _parenthesize_if_binop(node.left, left)
            if (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and float(node.right.value).is_integer()
            ):
                exponent = str(int(node.right.value))
                return f"{left_render}^{exponent}"
            right_render = _render_expression_pretty(node.right, name_map=name_map)
            return f"{left_render}^{_parenthesize_if_binop(node.right, right_render)}"
        if isinstance(node.op, ast.Mod):
            return (
                f"{_parenthesize_if_binop(node.left, left)} mod "
                f"{_parenthesize_if_binop(node.right, right)}"
            )
        return f"{left} ? {right}"

    if isinstance(node, ast.UnaryOp):
        operand = _render_expression_pretty(node.operand, name_map=name_map)
        operand = _parenthesize_if_binop(node.operand, operand)
        if isinstance(node.op, ast.USub):
            return f"-{operand}"
        if isinstance(node.op, ast.UAdd):
            return f"+{operand}"
        return operand

    if isinstance(node, ast.Call):
        call_name = _ast_callable_name(node.func)
        short_name = call_name.split(".")[-1] if call_name else ""
        display_name = _DISPLAY_FUNCTION_NAMES.get(
            call_name, _DISPLAY_FUNCTION_NAMES.get(short_name, short_name or "f")
        )
        args = [_render_expression_pretty(arg, name_map=name_map) for arg in node.args]
        if display_name == "abs" and len(args) == 1:
            return f"|{args[0]}|"
        return f"{display_name}({', '.join(args)})"

    if isinstance(node, ast.Attribute):
        return node.attr

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return "pi"
        if node.id == "e":
            return "e"
        if name_map:
            mapped = name_map.get(node.id)
            if mapped:
                return str(mapped)
        return node.id

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            return repr(value)
        return _format_number_literal(value)

    try:
        return ast.unparse(node)
    except Exception:
        return str(node)


def format_expression_pretty(
    expression_text: str,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    text = str(expression_text).strip()
    if not text:
        return ""
    try:
        tree = ast.parse(text, mode="eval")
        return _render_expression_pretty(tree.body, name_map=name_map)
    except Exception:
        fallback = text
        fallback = re.sub(r"\b(?:np|math)\.pi\b", "pi", fallback)
        fallback = re.sub(r"\b(?:np|math)\.e\b", "e", fallback)
        fallback = re.sub(
            r"\b(?:np|math)\.(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|exp|log10|log|sqrt|abs|power|minimum|maximum|clip)\b",
            lambda m: _DISPLAY_FUNCTION_NAMES.get(m.group(1), m.group(1)),
            fallback,
        )
        fallback = re.sub(r"\s*\*\s*", " · ", fallback)
        if name_map:
            for name, symbol in sorted(
                name_map.items(),
                key=lambda item: len(str(item[0])),
                reverse=True,
            ):
                raw_name = str(name).strip()
                rendered_symbol = str(symbol).strip()
                if not raw_name or not rendered_symbol or raw_name == rendered_symbol:
                    continue
                fallback = re.sub(
                    rf"\b{re.escape(raw_name)}\b", rendered_symbol, fallback
                )
        return re.sub(r"\s+", " ", fallback).strip()


def format_equation_pretty(
    equation_text: str,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    text = str(equation_text).strip()
    if not text:
        return ""
    if "=" not in text:
        return format_expression_pretty(text, name_map=name_map)
    lhs, rhs = text.split("=", 1)
    return f"{lhs.strip()} = {format_expression_pretty(rhs.strip(), name_map=name_map)}"


class ExpressionSyntaxHighlighter(QSyntaxHighlighter):
    """Colorize columns, parameters, and constants in the expression editor."""

    _WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    _NUMBER_RE = re.compile(r"(?<![A-Za-z_])(?:\d+\.\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?")

    def __init__(self, document):
        super().__init__(document)
        self.column_names = set()
        self.param_names = set()

        self.column_format = QTextCharFormat()
        self.column_format.setForeground(QColor(_EXPRESSION_COLUMN_COLOR))
        self.column_format.setFontWeight(QFont.Weight.Bold)

        self.param_format = QTextCharFormat()
        self.param_format.setForeground(QColor(_EXPRESSION_PARAM_COLOR))
        self.param_format.setFontWeight(QFont.Weight.Bold)

        self.constant_format = QTextCharFormat()
        self.constant_format.setForeground(QColor(_EXPRESSION_CONSTANT_COLOR))

    def set_context(self, column_names: Sequence[str], param_names: Sequence[str]):
        self.column_names = {str(name) for name in (column_names or [])}
        self.param_names = {str(name) for name in (param_names or [])}
        self.rehighlight()

    def highlightBlock(self, text):
        if not text:
            return

        for match in self._NUMBER_RE.finditer(text):
            start = match.start()
            end = match.end()
            self.setFormat(start, end - start, self.constant_format)

        for match in self._WORD_RE.finditer(text):
            token = match.group(0)
            start = match.start()
            end = match.end()
            if token in self.column_names:
                self.setFormat(start, end - start, self.column_format)
            elif token in self.param_names:
                self.setFormat(start, end - start, self.param_format)
            elif token in _EXPRESSION_ALLOWED_CONSTANTS:
                self.setFormat(start, end - start, self.constant_format)


class _ExpressionParameterCollector(ast.NodeVisitor):
    """Collect user-defined parameter names in first-seen order."""

    def __init__(self, reserved_names=None):
        super().__init__()
        self.names = []
        self._seen = set()
        self.reserved_names = set(reserved_names or ())

    def visit_Call(self, node):
        # Function tokens are validated separately; only parse argument expressions.
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        if name in self.reserved_names:
            return
        if name in _EXPRESSION_ALLOWED_FUNCTIONS:
            return
        if name in _EXPRESSION_ALLOWED_CONSTANTS:
            return
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)


def extract_expression_parameter_names(
    expression_text: str,
    reserved_names: Optional[Sequence[str]] = None,
) -> List[str]:
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Function expression is empty.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid function expression: {exc.msg}") from exc

    reserved = set(reserved_names or ())
    reserved |= {"np"} | _EXPRESSION_HELPER_NAMES
    collector = _ExpressionParameterCollector(reserved_names=reserved)
    collector.visit(tree)

    if not collector.names:
        raise ValueError("Function must reference at least one fit parameter.")

    for name in collector.names:
        if name == "x":
            raise ValueError(
                "Bare 'x' is not supported. Use explicit CSV columns (for example CH3 or TIME)."
            )
        if not _PARAMETER_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid parameter name '{name}' in expression.")
    return collector.names


def compile_expression_function(
    expression_text: str,
    parameter_names: Sequence[str],
):
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Function expression is empty.")

    ordered_names = list(parameter_names)
    if not ordered_names:
        raise ValueError("No parameters are defined for this function.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid function expression: {exc.msg}") from exc

    code = compile(tree, "<fit_expression>", "eval")
    eval_globals = {
        "__builtins__": __builtins__,
        "np": np,
        "math": math,
        **_EXPRESSION_ALLOWED_FUNCTIONS,
        **_EXPRESSION_ALLOWED_CONSTANTS,
    }

    def _prepare_channel_array(values, target_length: int):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == target_length:
            return array
        if array.size == 1:
            return np.full(target_length, float(array[0]), dtype=float)
        raise ValueError(
            f"Column length {array.size} does not match input length {target_length}."
        )

    def _evaluate(
        x_data,
        param_values,
        column_data=None,
    ):
        input_array = np.asarray(x_data, dtype=float).reshape(-1)
        n_points = input_array.size
        columns = {}
        if column_data:
            for name, values in column_data.items():
                try:
                    columns[str(name)] = _prepare_channel_array(values, n_points)
                except Exception:
                    continue

        def col(name):
            key = str(name)
            if key in columns:
                return columns[key]
            if key.upper() in columns:
                return columns[key.upper()]
            if key.lower() in columns:
                return columns[key.lower()]
            raise KeyError(f"Column '{key}' not found.")

        eval_locals = {
            "col": col,
            "columns": columns,
            "C": columns,
        }
        if "TIME" in columns:
            eval_locals["TIME"] = columns["TIME"]

        for key, values in columns.items():
            if _PARAMETER_NAME_RE.fullmatch(key):
                eval_locals[key] = values

        for name in ordered_names:
            if name not in param_values:
                raise ValueError(f"Missing parameter '{name}' for expression.")
            eval_locals[name] = float(param_values[name])

        try:
            result = eval(code, eval_globals, eval_locals)
        except Exception as exc:
            raise ValueError(f"Function evaluation failed: {exc}") from exc

        result_array = np.asarray(result, dtype=float)
        if result_array.shape == ():
            return np.full_like(input_array, float(result_array), dtype=float)
        result_array = result_array.reshape(-1)
        if result_array.size != n_points:
            raise ValueError("Function output length does not match input length.")
        return result_array

    return _evaluate
