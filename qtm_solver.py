import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Backend: Dual Formulator ---
def formulate_dual(num_vars, var_restrictions, constraints, obj_coeffs, obj_type):
    dual_obj_type = 'Min' if obj_type == 'Max' else 'Max'
    dual_vars = [f"y_{i+1}" for i in range(len(constraints))]
    
    obj_terms = []
    for i, c in enumerate(constraints):
        if c['rhs'] != 0:
            obj_terms.append(f"{c['rhs']}{dual_vars[i]}")
    
    dual_obj_str = " + ".join(obj_terms).replace("+ -", "- ") if obj_terms else "0"
    dual_obj_eq = f"\\text{{{dual_obj_type}}} W = {dual_obj_str}"
    
    dual_constraints_eqs = []
    for j in range(num_vars):
        terms = []
        for i, c in enumerate(constraints):
            coeff = c['coeffs'][j]
            if coeff != 0:
                terms.append(f"{coeff}{dual_vars[i]}")
        
        lhs = " + ".join(terms).replace("+ -", "- ") if terms else "0"
        
        # If primal var is unrestricted, dual constraint is an equality
        if var_restrictions[j] == 'Unrestricted':
            sign = "="
        else:
            sign = "\\ge" if obj_type == 'Max' else "\\le"
            
        rhs = obj_coeffs[j]
        dual_constraints_eqs.append(f"{lhs} {sign} {rhs}")
        
    restrictions = []
    for i, c in enumerate(constraints):
        if obj_type == 'Max':
            if c['type'] == '<=': restrictions.append(f"{dual_vars[i]} \\ge 0")
            elif c['type'] == '>=': restrictions.append(f"{dual_vars[i]} \\le 0")
            elif c['type'] == '=': restrictions.append(f"{dual_vars[i]} \\text{{ is unrestricted}}")
        else: 
            if c['type'] == '>=': restrictions.append(f"{dual_vars[i]} \\ge 0")
            elif c['type'] == '<=': restrictions.append(f"{dual_vars[i]} \\le 0")
            elif c['type'] == '=': restrictions.append(f"{dual_vars[i]} \\text{{ is unrestricted}}")

    return dual_obj_eq, dual_constraints_eqs, restrictions

# --- 2. Backend: Plotting ---
def plot_lpp(constraints, obj_coeffs, obj_type):
    if len(obj_coeffs) != 2: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    x_max, y_max = 0, 0
    
    for c in constraints:
        a, b = c['coeffs']
        rhs = c['rhs']
        if a != 0: x_max = max(x_max, rhs/a if rhs!=0 else 10)
        if b != 0: y_max = max(y_max, rhs/b if rhs!=0 else 10)
        
    x_lim = x_max * 1.2 if x_max != 0 else 10
    y_lim = y_max * 1.2 if y_max != 0 else 10
    d = np.linspace(0, x_lim, 400)
    x, y = np.meshgrid(d, d)
    
    for c in constraints:
        a, b, rhs, type_ = c['coeffs'][0], c['coeffs'][1], c['rhs'], c['type']
        x_vals = np.linspace(0, x_lim, 100)
        if b != 0:
            ax.plot(x_vals, (rhs - a*x_vals) / b, label=f'{a}x1 + {b}x2 {type_} {rhs}')
        else:
            ax.axvline(x=rhs/a, label=f'{a}x1 {type_} {rhs}')
            
    feasible_grid = np.ones(x.shape, dtype=bool)
    for c in constraints:
        a, b, rhs, type_ = c['coeffs'][0], c['coeffs'][1], c['rhs'], c['type']
        if type_ == '<=': feasible_grid &= (a*x + b*y <= rhs)
        elif type_ == '>=': feasible_grid &= (a*x + b*y >= rhs)
        elif type_ == '=': feasible_grid &= (np.abs(a*x + b*y - rhs) < 0.1)
            
    feasible_grid &= (x >= 0) & (y >= 0)
    ax.imshow(feasible_grid.astype(int), extent=(0, x_lim, 0, x_lim), origin='lower', cmap='Greys', alpha=0.3)
    color = 'blue' if obj_type == 'Min' else 'red'
    ax.quiver(0, 0, obj_coeffs[0], obj_coeffs[1], color=color, scale=1, scale_units='xy', label='Obj Gradient')
    
    ax.set_xlim(0, x_lim); ax.set_ylim(0, y_lim)
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.grid(True); ax.legend(); ax.set_title("Feasible Region")
    return fig

# --- 3. Backend: Solver Logic ---
def solve_linear_program(num_vars_original, var_restrictions, constraints, obj_coeffs_orig, obj_type='Max'):
    col_names = []
    objective_coeffs = []
    var_map = {} 
    
    col_idx = 0
    for i in range(num_vars_original):
        if var_restrictions[i] == 'Unrestricted':
            col_names.extend([f'x{i+1}^+', f'x{i+1}^-'])
            objective_coeffs.extend([obj_coeffs_orig[i], -obj_coeffs_orig[i]])
            var_map[i] = (col_idx, col_idx + 1)
            col_idx += 2
        else:
            col_names.append(f'x{i+1}')
            objective_coeffs.append(obj_coeffs_orig[i])
            var_map[i] = (col_idx, None)
            col_idx += 1
            
    num_vars = len(col_names)
    cj = [-1 * c for c in objective_coeffs] if obj_type == 'Min' else list(objective_coeffs)
    num_constraints = len(constraints)
    M = 10000.0  
    
    slack_count = artificial_count = 0
    initial_basic_cols = [] 
    
    for c_data in constraints:
        if c_data['type'] in ['<=', '>=']:
            col_names.append(f's{slack_count+1}')
            cj.append(0)
            slack_count += 1
    for c_data in constraints:
        if c_data['type'] in ['>=', '=']:
            col_names.append(f'A{artificial_count+1}')
            cj.append(-M) 
            artificial_count += 1

    matrix, basic_vars_idx = [], []
    slack_ptr = artif_ptr = 0
    
    for i, c_data in enumerate(constraints):
        orig_row = list(c_data['coeffs'])
        row = []
        for j in range(num_vars_original):
            if var_restrictions[j] == 'Unrestricted': row.extend([orig_row[j], -orig_row[j]])
            else: row.append(orig_row[j])
                
        slack_part = [0] * slack_count
        artif_part = [0] * artificial_count
        
        if c_data['type'] == '<=':
            slack_part[slack_ptr] = 1
            basic_col = num_vars + slack_ptr
            basic_vars_idx.append(basic_col) 
            initial_basic_cols.append(basic_col)
            slack_ptr += 1
        elif c_data['type'] == '>=':
            slack_part[slack_ptr] = -1
            slack_ptr += 1
            
        if c_data['type'] in ['>=', '=']:
            artif_part[artif_ptr] = 1
            basic_col = num_vars + slack_count + artif_ptr
            basic_vars_idx.append(basic_col)
            initial_basic_cols.append(basic_col)
            artif_ptr += 1
            
        row.extend(slack_part + artif_part + [c_data['rhs']])
        matrix.append(row)

    tableau = np.array(matrix, dtype=float)
    cj = np.array(cj, dtype=float)
    steps = []
    status = "In Progress"
    
    for it in range(20):
        cb = cj[basic_vars_idx]
        zj = np.dot(cb, tableau[:, :-1])
        net_eval = zj - cj 
        
        df = pd.DataFrame(tableau, columns=col_names + ['Sol'])
        df.insert(0, 'Basic Var', [col_names[i] for i in basic_vars_idx])
        df.loc['Zj - Cj'] = [''] + list(net_eval) + [np.nan]
        steps.append(df)
        
        if np.all(net_eval >= -1e-5):
            status = "Optimal"
            break
            
        entering_col = np.argmin(net_eval)
        if np.all(tableau[:, entering_col] <= 0):
            status = "Unbounded"; break
            
        ratios = [tableau[i, -1] / tableau[i, entering_col] if tableau[i, entering_col] > 1e-5 else np.inf for i in range(num_constraints)]
        leaving_row = np.argmin(ratios)
        if ratios[leaving_row] == np.inf:
             status = "Unbounded"; break

        pivot_val = tableau[leaving_row, entering_col]
        basic_vars_idx[leaving_row] = entering_col
        tableau[leaving_row, :] /= pivot_val
        for i in range(num_constraints):
            if i != leaving_row: tableau[i, :] -= tableau[i, entering_col] * tableau[leaving_row, :]
    else:
        status = "Max Iterations Reached"

    if status != "Optimal":
         # Fixed return bug here!
         return steps, status, 0, {}, {}, {}, {}

    final_z = np.dot(cj[basic_vars_idx], tableau[:, -1]) * (-1 if obj_type == 'Min' else 1)
        
    final_vars = {}
    for i in range(num_vars_original):
        idx_plus, idx_minus = var_map[i]
        val_plus = tableau[basic_vars_idx.index(idx_plus), -1] if idx_plus in basic_vars_idx else 0.0
        val_minus = tableau[basic_vars_idx.index(idx_minus), -1] if idx_minus and idx_minus in basic_vars_idx else 0.0
        final_vars[f'x{i+1}'] = val_plus - val_minus

    final_net_eval = steps[-1].loc['Zj - Cj'].values[1:-1]
    shadow_prices, range_feasibility, range_optimality = {}, {}, {}
    
    for i in range(num_constraints):
        orig_rhs = constraints[i]['rhs']
        b_inv_col_idx = initial_basic_cols[i]
        shadow_prices[f'C{i+1}'] = final_net_eval[b_inv_col_idx]
        
        lower_deltas = [-tableau[r, -1] / tableau[r, b_inv_col_idx] for r in range(num_constraints) if tableau[r, b_inv_col_idx] > 1e-5]
        upper_deltas = [-tableau[r, -1] / tableau[r, b_inv_col_idx] for r in range(num_constraints) if tableau[r, b_inv_col_idx] < -1e-5]
        
        delta_down = max(lower_deltas) if lower_deltas else -np.inf
        delta_up = min(upper_deltas) if upper_deltas else np.inf
        range_feasibility[f'C{i+1} RHS'] = {'Current': orig_rhs, 'Allowable Min': orig_rhs + delta_down if delta_down != -np.inf else "-∞", 'Allowable Max': orig_rhs + delta_up if delta_up != np.inf else "∞"}

    for i in range(num_vars_original):
        idx_plus, _ = var_map[i] 
        orig_cj = obj_coeffs_orig[i]
        if idx_plus not in basic_vars_idx:
             reduced_cost = final_net_eval[idx_plus]
             r_min, r_max = ("-∞", orig_cj + reduced_cost) if obj_type == 'Max' else (orig_cj - reduced_cost, "∞")
        else:
             row_vals = tableau[basic_vars_idx.index(idx_plus), :-1]
             upper_deltas = [final_net_eval[k] / row_vals[k] for k in range(len(cj)) if k not in basic_vars_idx and row_vals[k] > 1e-5]
             lower_deltas = [final_net_eval[k] / row_vals[k] for k in range(len(cj)) if k not in basic_vars_idx and row_vals[k] < -1e-5]
             delta_down = max(lower_deltas) if lower_deltas else -np.inf
             delta_up = min(upper_deltas) if upper_deltas else np.inf
             r_min = orig_cj + delta_down if delta_down != -np.inf else "-∞"
             r_max = orig_cj + delta_up if delta_up != np.inf else "∞"
             
        range_optimality[f'x{i+1}'] = {'Current': orig_cj, 'Allowable Min': round(r_min, 4) if isinstance(r_min, float) else r_min, 'Allowable Max': round(r_max, 4) if isinstance(r_max, float) else r_max}

    return steps, status, final_z, final_vars, shadow_prices, range_optimality, range_feasibility

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Advanced QTM Solver", layout="wide")
st.title("📈 Advanced QTM Linear Programming Solver")
st.markdown("Supports **Unrestricted Variables**, **Dual Formulation**, **Graphical Visuals**, and **Sensitivity Analysis**.")

with st.sidebar:
    st.header("Settings")
    obj_type = st.radio("Objective", ["Max", "Min"])
    num_vars = st.number_input("Variables (x)", 1, 10, 2)
    num_const = st.number_input("Constraints", 1, 10, 2)

st.subheader("1. Define Problem")
st.markdown("#### Objective Function & Variable Types")
cols_obj = st.columns(num_vars)
obj_coeffs, var_restrictions = [], []

for i in range(num_vars):
    val = cols_obj[i].number_input(f"Coeff x{i+1}", value=0.0, step=1.0, key=f"obj_{i}")
    restr = cols_obj[i].selectbox("Restriction", [">= 0", "Unrestricted"], key=f"res_{i}")
    obj_coeffs.append(val); var_restrictions.append(restr)

st.divider()
st.markdown("#### Constraints")
constraints = []
header_cols = st.columns(num_vars + 2)
for j in range(num_vars): header_cols[j].markdown(f"**Coeff x{j+1}**")
header_cols[num_vars].markdown("**Type**")
header_cols[num_vars+1].markdown("**RHS**")

for i in range(num_const):
    cols = st.columns(num_vars + 2)
    row_coeffs = []
    for j in range(num_vars):
        row_coeffs.append(cols[j].number_input(f"a{i+1}{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}", label_visibility="collapsed"))
    ctype = cols[num_vars].selectbox("Type", ["<=", ">=", "="], key=f"type_{i}", label_visibility="collapsed")
    rhs = cols[num_vars+1].number_input("RHS", value=0.0, step=1.0, key=f"rhs_{i}", label_visibility="collapsed")
    constraints.append({'coeffs': row_coeffs, 'type': ctype, 'rhs': rhs})

st.divider()

if st.button("Solve, Visualize & Analyze", type="primary"):
    
    st.subheader("2. Dual Problem Formulation")
    dual_obj, dual_consts, dual_restr = formulate_dual(num_vars, var_restrictions, constraints, obj_coeffs, obj_type)
    st.latex(dual_obj)
    st.markdown("**Subject to:**")
    for eq in dual_consts: st.latex(eq)
    st.markdown("**Where:**")
    for r in dual_restr: st.latex(r)
    st.divider()
    
    if num_vars == 2:
        st.subheader("3. Graphical Visualization")
        fig = plot_lpp(constraints, obj_coeffs, obj_type)
        if fig: st.pyplot(fig)
        st.divider()
    
    st.subheader("4. Iterations (Tableau)")
    try:
        steps, status, z_val, vars_val, shadow_prices, r_opt, r_feas = solve_linear_program(num_vars, var_restrictions, constraints, obj_coeffs, obj_type)
        
        if status != "Optimal":
            st.error(f"Solver Status: {status}")
        else:
            tabs = st.tabs([f"Iteration {i}" for i in range(len(steps))])
            for i, tab in enumerate(tabs):
                with tab:
                    df = steps[i]
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.dataframe(df.style.format(subset=numeric_cols, formatter="{:.2f}").highlight_min(axis=0, subset=pd.IndexSlice['Zj - Cj', numeric_cols[:-1]], color='#ffcccb'))
            
            st.divider()
            st.subheader("5. Final Result")
            c1, c2 = st.columns(2)
            c1.success(f"**Optimal Z ({obj_type}) = {z_val:.4f}**")
            c2.write("**Decision Variables:**"); c2.json(vars_val)
            
            st.divider()
            st.subheader("6. Sensitivity Analysis")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("##### Range of Optimality (Cost Coeffs)")
                st.table(pd.DataFrame.from_dict(r_opt, orient='index'))
            with s2:
                st.markdown("##### Range of Feasibility (RHS)")
                st.table(pd.DataFrame.from_dict(r_feas, orient='index'))
            
            export_df = pd.concat([df.assign(Iteration=idx) for idx, df in enumerate(steps)])
            st.download_button(label="📥 Download Tableaus as CSV", data=export_df.to_csv(index=False).encode('utf-8'), file_name='qtm_solution.csv', mime='text/csv')
                
    except Exception as e:
        st.error(f"Error during calculation: {e}")
