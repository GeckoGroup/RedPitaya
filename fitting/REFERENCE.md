# Fitting GUI — Developer Reference

Quick-reference for anyone extending the codebase. Covers the procedure system
architecture and the visual design system.

---

## 1. Procedure System Architecture

### 1.1 Module Layout

```
procedure_steps.py   Step data types + registry (pure logic, no Qt)
procedure.py         Pipeline engine + FitProcedure container
procedure_widgets.py ProcedurePanel GUI (self-contained QWidget)
batch.py             ProcedureFitWorker, BatchProcedureFitWorker (Qt workers)
fit_gui.py           Hosts ProcedurePanel via ProcedureHost adapter
```

### 1.2 Step Type System

Steps are **frozen dataclasses** that inherit `ProcedureStepBase` and register
themselves with `@register_step_type`. Each step has:

| Attribute        | Purpose |
|------------------|---------|
| `step_type`      | `ClassVar[str]` — unique key for (de)serialisation |
| `step_label`     | `ClassVar[str]` — human-readable name for the UI |
| `serialize()`    | Returns a JSON-safe `dict` with `"step_type"` key |
| `deserialize(d)` | Classmethod — constructs the step from a dict |
| `execute(ctx)`   | Runs the step, mutates `ProcedureContext`, returns `StepResult` |

**Current step types:**

| `step_type`         | Class                 | Purpose |
|---------------------|-----------------------|---------|
| `fit`               | `FitStep`             | Run curve-fitting with channel/param selection, R² gate, retries |
| `set_parameter`     | `SetParameterStep`    | Seed parameters from literal/parameter/capture sources with scale+offset transforms |
| `set_boundaries`    | `SetBoundariesStep`   | Set named boundary-group ratios from literals, other groups, or expressions |
| `randomize_seeds`   | `RandomizeSeedsStep`  | Perturb parameter seeds within bounds |

### 1.3 Adding a New Step Type

1. **Define the dataclass** in `procedure_steps.py`:

```python
@register_step_type
@dataclass(frozen=True)
class MyNewStep(ProcedureStepBase):
    step_type: ClassVar[str] = "my_new_step"
    step_label: ClassVar[str] = "My New Step"

    # Add any fields specific to this step...
    some_option: str = ""

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        if self.some_option:
            d["some_option"] = self.some_option
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> "MyNewStep":
        return cls(
            some_option=str(data.get("some_option") or ""),
            label=str(data.get("label") or ""),
        )

    def execute(self, context: ProcedureContext) -> StepResult:
        # Mutate context.seed_map, context.boundary_seeds, etc.
        # ...
        return StepResult(
            status="pass",
            message="Did the thing.",
            params_by_key=dict(context.seed_map),
        )
```

2. **Import the class** in `procedure_widgets.py` so the UI can offer it in the
   "Add step" menu. The `available_step_types()` function reads the registry
   automatically — no manual menu wiring needed.

3. **Optionally add a custom card builder** in `ProcedurePanel._make_step_card()`
   if the step needs specialised UI controls beyond the label.

That's it — serialisation, pipeline execution, and batch processing all pick up
the new type automatically.

### 1.4 Execution Flow

```
fit_gui.py  ManualFitGUI
   │
   ├─ _make_procedure_host()  → creates ProcedureHost adapter (inner class)
   │                              delegates to GUI methods for data/model access
   │
   └─ ProcedurePanel  (procedure_widgets.py)
        │
        ├─ build_procedure()   → FitProcedure(steps=...)
        ├─ _run_procedure()    → spawns ProcedureFitWorker on QThread
        │
        └─ ProcedureFitWorker  (batch.py)
             │
             └─ run_procedure_pipeline()  (procedure.py)
                  │
                  └─ for step in procedure.steps:
                       step.execute(context)   ← polymorphic dispatch
                       │
                       └─ FitStep.execute() calls _execute_fit_step()
                            which calls model.run_multi_channel_fit_pipeline()
                            or model.run_piecewise_fit_pipeline()
```

### 1.5 Key Design Decisions

- **ProcedureHost protocol** — The procedure panel never imports `ManualFitGUI`
  directly. Instead it talks to a `ProcedureHost` interface. This means the
  panel is testable in isolation and could be embedded in a different host.

- **Lazy imports** — `procedure.py` lazily imports `model` to break the
  circular dependency (`model → solver`, `procedure → model`,
  `procedure_steps → procedure`). `FitStep.execute()` does a lazy
  `from procedure import _execute_fit_step`.

- **Frozen dataclasses** — Steps are immutable value objects. When the user edits
  a step in the UI, a new instance is created and replaces the old one in the
  list. This keeps serialisation trivial and avoids accidental mutation.

- **Mutable context** — `ProcedureContext` is the *one* mutable object. Steps
  read and write `context.seed_map`, `context.boundary_seeds`, etc. This is
  how results feed forward between steps.

- **StepResult** — Also a frozen dataclass. The pipeline collects these and
  reports them via `step_callback` for live procedure progress updates.

### 1.6 Serialisation Format

```json
{
  "name": "My Procedure",
  "steps": [
    {
      "step_type": "fit",
      "label": "Initial broad fit",
      "free_params": ["A", "B"],
      "min_r2": 0.9,
      "max_retries": 3
    },
    {
      "step_type": "set_parameter",
      "assignments": [
        {
          "target_key": "f_mod",
          "source_kind": "capture",
          "source_key": "f_mod",
          "scale": 1.0,
          "offset": 0.0
        }
      ]
    },
    {
      "step_type": "fit",
      "label": "Fine tune",
      "fixed_params": ["C"]
    }
  ]
}
```

Procedure payloads must include `"step_type"` on every step. Any step without
that key is rejected at load time.

### 1.7 Batch Integration

`BatchProcedureFitWorker` iterates over files, loads data,
resolves capture-field mappings, and calls `run_procedure_pipeline()` for
each file. It emits the same `progress(int, int, object)` signal as
`BatchFitWorker`, so the GUI reuses the same result-handling code.

---

## 2. Visual Design System

### 2.1 Theme Foundation

- **Style engine:** Qt Fusion, light-mode only
- **Palette inspiration:** Tailwind CSS slate/gray + blue accent
- **Enforced in:** `_enforce_light_mode()` sets `QPalette` globally

### 2.2 Core Colour Palette

#### Neutrals

| Token           | Hex       | Usage |
|-----------------|-----------|-------|
| `bg-root`       | `#f5f7fa` | Window background, disabled inputs |
| `bg-card`       | `#ffffff` | Cards, inputs, buttons, selected tabs |
| `bg-alt`        | `#f8fafc` | Alternate table rows, scroll area bg |
| `bg-hover`      | `#f3f6f9` | Button hover, read-only input, table header |
| `bg-pressed`    | `#eaf0f5` | Button pressed |
| `bg-tab`        | `#eef2f6` | Unselected tab background |

#### Text

| Token           | Hex       | Weight | Usage |
|-----------------|-----------|--------|-------|
| `text-primary`  | `#111827` | —      | Body text, input text |
| `text-heading`  | `#0f172a` | 700    | Prominent headings, param value labels |
| `text-label`    | `#334155` | 600    | Section/field labels |
| `text-secondary`| `#374151` | —      | Headers, inline param labels |
| `text-muted`    | `#64748b` | —      | Hints, counts, param headers |
| `text-faint`    | `#6b7280` | —      | Column tokens, type tags |
| `text-disabled` | `#9ca3af` | —      | Disabled state |

#### Accent (blue)

| Token           | Hex       | Usage |
|-----------------|-----------|-------|
| `blue-600`      | `#2563eb` | Primary buttons, checkboxes, status, step numbers |
| `blue-700`      | `#1d4ed8` | Primary hover, source paths, active slider |
| `blue-800`      | `#1e40af` | Primary pressed |
| `blue-100`      | `#dbeafe` | Checked button bg, selection highlight |
| `blue-200`      | `#bfdbfe` | Token button border |
| `blue-300`      | `#93c5fd` | Checked button border |

#### Status

| Meaning  | Text        | Background  |
|----------|-------------|-------------|
| Success  | `#15803d`   | `#dcfce7`   |
| Error    | `#dc2626`   | `#fee2e2`   |
| Running  | `#7c3aed`   | —           |
| Skipped  | —           | `#f1f5f9`   |

#### Borders

| Token           | Hex       | Usage |
|-----------------|-----------|-------|
| `border-card`   | `#e3e8ef` | GroupBox, tab pane, toolbar, scroll area |
| `border-input`  | `#d3dae3` | Buttons, inputs, tables, step cards |
| `border-tab`    | `#d7dde6` | Unselected tab |
| `border-grid`   | `#e7ecf2` | Table gridlines |
| `border-hover`  | `#c7d0dc` | Button hover |
| `border-strong` | `#94a3b8` | Param bound/value spinboxes |
| `border-sep`    | `#e5e7eb` | Inline separator lines |

### 2.3 Spacing

| Value | Typical use |
|-------|-------------|
| `0`   | Zero-gap stacking (equation layouts) |
| `2px` | Compact rows (sliders, expression tokens, file list items) |
| `4px` | Default row/column spacing, main layout |
| `6px` | Tab layouts, header rows, step card inner spacing |
| `8px` | Panel splits, dialog spacing, step card scroll spacing |

**Content margins:**

| Value               | Where |
|---------------------|-------|
| `(6, 6, 6, 6)`     | Main layout, all tab layouts, cards |
| `(5, 5, 5, 5)`     | Parameters frame, fit options |
| `(0, 0, 0, 0)`     | Sub-layouts, inline rows, toolbars |
| `(8, 6, 8, 6)`     | Step card inner vlayout |
| `(10, 10, 10, 10)` | Dialogs |

### 2.4 Border Radii

| Value  | Where |
|--------|-------|
| `10px` | Procedure step cards |
| `8px`  | Cards (QGroupBox), buttons, tab pane, scroll areas |
| `6px`  | Inputs, toolbars, tables, tab headers, slider handle |
| `3px`  | Small checkboxes (inside step cards) |
| `2px`  | Slider groove |

### 2.5 Component Quick-Reference

#### Buttons

```
min-height: 22px;  padding: 2px 8px;
bg: #ffffff;  border: 1px solid #d3dae3;  border-radius: 8px;
hover:   bg #f3f6f9, border #c7d0dc
pressed: bg #eaf0f5
checked: bg #dbeafe, border #93c5fd
```

Primary variant: set `button.setProperty("primary", True)`.
→ bg `#2563eb`, text `white`, border `#1d4ed8`.

Destructive: inline style `color: #dc2626; hover bg: #fee2e2`.

#### Inputs (QLineEdit / QComboBox / QSpinBox)

```
min-height: 22px;  padding: 1px 6px;
bg: #ffffff;  border: 1px solid #d3dae3;  border-radius: 6px;
```

#### Cards (QGroupBox)

```
bg: #ffffff;  border: 1px solid #e3e8ef;  border-radius: 8px;
title: hidden (color: transparent; max-height: 0px)
```

Step cards use `border-radius: 10px` and `border-color: #d3dae3`.

#### Tables

```
bg: #ffffff;  border: 1px solid #d3dae3;  border-radius: 6px;
gridline-color: #e7ecf2;  alternate-bg: #f8fafc;
header: bg #f3f6f9, text #374151, border #e2e8f0
```

#### Tabs

```
pane: border 1px solid #e3e8ef, border-radius 8px, bg #ffffff
tab:  bg #eef2f6, border #d7dde6, radius 6px top, padding 4px 10px
selected: bg #ffffff, text #111827
```

#### Checkboxes

```
spacing: 4px
indicator: 14×14px, border 2px solid #9ca3af, border-radius 3px, bg #ffffff
checked:  bg #2563eb, border #2563eb
hover:    border #6b7280
disabled: bg #f5f7fa, border #d3dae3
checked+disabled: bg #93c5fd, border #93c5fd
```

Applied globally via `_apply_compact_ui_defaults()`. Step cards inherit the
same style — no separate card-scoped override needed.

#### Scroll Areas

```
border: 1px solid #e3e8ef;  border-radius: 8px;  bg: #f8fafc
```

Override inline with `border: none` for embedded/borderless contexts
(e.g. parameter controls scroll area).

### 2.6 Object Names for QSS Targeting

| Name              | Widget           | Key styles |
|-------------------|------------------|------------|
| `root`            | Central widget   | Background `#f5f7fa` |
| `statusLabel`     | QLabel           | `color: #2563eb; font-weight: 600` |
| `sourcePathLabel` | QLabel           | `color: #1d4ed8; text-decoration: underline` |
| `columnTokenLabel`| QLabel           | `color: #6b7280; font-weight: 600` |
| `paramInline`     | QLabel           | `color: #374151` |
| `paramHeader`     | QLabel           | `color: #6b7280; font-size: 11px; font-weight: 600` |
| `paramBoundBox`   | QDoubleSpinBox   | `border-color: #94a3b8; color: #0f172a` |
| `paramValueBox`   | QDoubleSpinBox   | Same + `font-weight: 600` |
| `procStepCard`    | QGroupBox        | Step card styling |
| `statsLine`       | QLabel           | Stats display |

### 2.7 Widget Factory Methods

Use these instead of constructing widgets manually — they apply consistent
styling automatically.

```python
# In ManualFitGUI (fit_gui.py):
self._new_label(text, object_name=..., tooltip=..., width=..., alignment=..., style_sheet=..., word_wrap=...)
self._new_button(text, handler=..., primary=..., fixed_width=..., tooltip=..., style_sheet=...)
self._new_checkbox(text, checked=..., tooltip=..., toggled_handler=...)
self._new_combobox(items=..., minimum_width=..., rich_text=..., current_index_changed=...)
self._new_line_edit(text, placeholder=..., read_only=..., fixed_width=..., text_changed=...)

# In ProcedurePanel (procedure_widgets.py), static equivalents:
ProcedurePanel._make_label(...)
ProcedurePanel._make_button(...)
```

### 2.8 Chart / Plot Colours

**Channel palette** (in order):
`#2563eb` `#f59e0b` `#7c3aed` `#dc2626` `#0891b2`
`#ea580c` `#0f766e` `#a855f7` `#64748b` `#be123c`

**Fit curve overlay:** `#16a34a` (green-600)

Access via `palette_color(index)` and `FIT_CURVE_COLOR` from `model.py`.

---

## 3. Conventions

### 3.1 File Organisation

| File | Responsibility |
|------|---------------|
| `model.py` | Data types, fitting pipelines, model definitions — **no Qt** |
| `solver.py` | Numeric solver (piecewise fitting, grid search) — **no Qt** |
| `expression.py` | Expression parsing, segment specs — **no Qt** |
| `procedure_steps.py` | Step types + registry — **no Qt** |
| `procedure.py` | Pipeline execution — **no Qt** |
| `batch.py` | QObject workers (threaded) — minimal Qt (signals/slots only) |
| `procedure_widgets.py` | Procedure tab UI — full Qt |
| `widgets.py` | Reusable widget classes — full Qt |
| `fit_gui.py` | Main GUI window — full Qt, orchestrates everything |
| `data_io.py` | File I/O utilities — **no Qt** |

### 3.2 Patterns to Follow

- **Frozen dataclasses** for any model/config/step type.
- **Protocol classes** (like `ProcedureHost`) for decoupling GUI panels from
  the main window — the panel only sees the protocol, never the concrete class.
- **QThread + QObject worker** pattern for background tasks. The worker lives
  on the thread, communicates via signals. See `ProcedureFitWorker` for the
  template.
- **`cancel_check` callback** for cooperative cancellation — functions accept
  `cancel_check: Callable[[], bool]` and poll it periodically.
- **Factory methods** (`_new_label`, `_new_button`, etc.) for consistent widget
  creation — never construct bare `QPushButton()` in layout code.

### 3.3 Anti-Patterns to Avoid

- **Don't import `ManualFitGUI` from sub-modules.** Use protocol/adapter instead.
- **Don't store mutable state in dataclass fields.** Use tuples, not lists.
- **Don't inline large stylesheets** into widget constructors — use object names
  and the app-level stylesheet in `_apply_global_stylesheet()`.
- **Don't create circular imports.** If module A needs module B at call time
  only, use a lazy import inside the function body.
