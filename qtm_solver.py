import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. Backend: Dual Formulator ---
def formulate_dual(num_vars, constraints, obj_coeffs, obj_type):
    dual_obj_type = 'Min' if obj_type == 'Max' else 'Max'
    
    # Dual variables y1, y2...
    dual_vars = [f"y_{i+1}" for i in range(len(constraints))]
    
    # Dual Objective Function
    obj_terms = []
    for i, c in enumerate(constraints):
        if c['rhs'] != 0:
            obj_terms.append(f"{c['rhs']}{dual_vars[i]}")
    
    if not obj_terms:
        dual_obj_str = "0"
    else:
        dual_obj_str = " + ".join(obj_terms).replace("+ -", "- ")
        
    dual_obj_eq = f"\\text{{{dual_obj_type}}} W = {dual_obj_str}"
    
    # Dual Constraints
    dual_constraints_eqs = []
    for j in range(num_vars):
        terms = []
        for i, c in enumerate(constraints):
            coeff = c['coeffs'][j]
            if coeff != 0:
                terms.append(f"{coeff}{dual_vars[i]}")
        
        if not terms:
            lhs = "0"
        else:
            lhs = " + ".join(terms).replace("+ -", "- ")
            
        # Determine sign based on Primal Obj Type and Primal Variable (assuming x_j >= 0)
        sign = "\\ge" if obj_type == 'Max' else "\\le"
        rhs = obj_coeffs[j]
        
        dual_constraints_eqs.append(f"{lhs} {sign} {rhs}")
        
    # Dual Variable Restrictions
    restrictions = []
    for i, c in enumerate(constraints):
        if obj_type == 'Max':
            if c['type'] == '<=': restrictions.append(f"{dual_vars[i]} \\ge 0")
            elif c['type'] == '>=': restrictions.append(f"{dual_vars[i]} \\le 0")
            elif c['type'] == '=': restrictions.append(f"{dual_vars[i]} \\text{{ is unrestricted}}")
        else: # Min
            if c['type'] == '>=': restrictions.append(f"{dual_vars[i]} \\ge 0")
            elif c['type'] == '<=': restrictions.append(f"{dual_vars[i]} \\le 0")
            elif c['type'] == '=': restrictions.append(f"{dual_vars[i]} \\text{{ is unrestricted}}")

    return dual_obj_eq, dual_constraints_eqs, restrictions

# --- 2. Backend: Solver Logic (Big M Method) ---
def solve_linear_program(num_vars, constraints, objective_coeffs, obj_type='Max'):
    
    if obj_type == 'Min':
        objective_coeffs = [-1 * c for c in objective_coeffs]
    
    num_constraints = len(constraints)
    M = 10000.0  
    
    col_names = [f'x{i+1}' for i in range(num_vars)]
    cj = list(objective_coeffs)
    
    slack_count = 0
    artificial_count = 0
    
    # Add Slacks/Surplus
    for c_data in constraints:
        if c_data['type'] == '<=':
            col_names.append(f's{slack_count+1}')
            cj.append(0)
            slack_count += 1
        elif c_data['type'] == '>=':
            col_names.append(f's{slack_count+1}')
            cj.append(0) 
            slack_count += 1
    
    # Add Artificials
    for c_data in constraints:
        if c_data['type'] in ['>=', '=']:
            col_names.append(f'A{artificial_count+1}')
            cj.append(-M) 
            artificial_count += 1

    matrix = [] 
    basic_vars_idx = [] 
    
    slack_ptr = 0
    artif_ptr = 0
    
    for i, c_data in enumerate(constraints):
        row = list(c_data['coeffs']) 
        
        slack_part = [0] * slack_count
        if c_data['type'] == '<=':
            slack_part[slack_ptr] = 1
            basic_vars_idx.append(num_vars + slack_ptr) 
            slack_ptr += 1
        elif c_data['type'] == '>=':
            slack_part[slack_ptr] = -1
            slack_ptr += 1
        
        row.extend(slack_part)
        
        artif_part = [0] * artificial_count
        if c_data['type'] in ['>=', '=']:
            artif_part[artif_ptr] = 1
            basic_vars_idx.append(num_vars + slack_count + artif_ptr)
            artif_ptr += 1
            
        row.extend(artif_part)
        row.append(c_data['rhs']) 
        
        matrix.append(row)

    tableau = np.array(matrix, dtype=float)
    cj = np.array(cj, dtype=float)
    steps = []
    
    max_iter = 20
    status = "In Progress"
    
    for it in range(max_iter):
        
        cb = cj[basic_vars_idx]
        body = tableau[:, :-1]
        
        # Calculate Zj and Zj - Cj
        zj = np.dot(cb, body)
        net_eval = zj - cj 
        
        df = pd.DataFrame(tableau, columns=col_names + ['Sol'])
        df.insert(0, 'Basic Var', [col_names[i] for i in basic_vars_idx])
        
        row_net = list(net_eval) + [np.nan]
        df.loc['Zj - Cj'] = [''] + row_net
        
        steps.append(df)
        
        # Optimality Check: For Maximize, all Zj - Cj >= 0
        if np.all(net_eval >= -1e-5):
            status = "Optimal"
            break
            
        # Entering Variable: Most negative Zj - Cj
        entering_col = np.argmin(net_eval)
        
        if np.all(tableau[:, entering_col] <= 0):
            status = "Unbounded"
            return steps, status, 0, {}, {}, {}
            
        # Ratio Test
        ratios = []
        for i in range(num_constraints):
            val = tableau[i, entering_col]
            sol = tableau[i, -1]
            if val > 1e-5:
                ratios.append(sol / val)
            else:
                ratios.append(np.inf)
        
        leaving_row = np.argmin(ratios)
        
        if ratios[leaving_row] == np.inf:
             status = "Unbounded"
             return steps, status, 0, {}, {}, {}

        # Pivot Operations
        pivot_val = tableau[leaving_row, entering_col]
        basic_vars_idx[leaving_row] = entering_col
        tableau[leaving_row, :] /= pivot_val
        
        for i in range(num_constraints):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i, :] -= factor * tableau[leaving_row, :]
    
    else:
        status = "Max Iterations Reached"

    # Extract Final Solution
    final_z = np.dot(cj[basic_vars_idx], tableau[:, -1])
    if obj_type == 'Min':
        final_z *= -1 
        
    final_vars = {}
    for i in range(num_vars):
        vname = f'x{i+1}'
        if (i) in basic_vars_idx:
            row_idx = basic_vars_idx.index(i)
            final_vars[vname] = tableau[row_idx, -1]
        else:
            final_vars[vname] = 0.0

    # Sensitivity Analysis Extraction
    shadow_prices = {}
    reduced_costs = {}
    
    if status == "Optimal":
        final_net_eval = steps[-1].loc['Zj - Cj'].values[1:-1] # Exclude Basic Var and Sol
        
        for i, name in enumerate(col_names):
            if name.startswith('x'):
                reduced_costs[name] = final_net_eval[i]
            elif name.startswith('s'):
                constraint_idx = int(name[1:])
                shadow_prices[f'Constraint {constraint_idx} RHS'] = final_net_eval[i]

    return steps, status, final_z, final_vars, shadow_prices, reduced_costs

# --- 3. Visualization Logic ---
def plot_lpp(constraints, obj_coeffs, obj_type):
    if len(obj_coeffs) != 2: return None
        
    fig, ax = plt.subplots(figsize=(8, 6))
    x_max = 0
    y_max = 0
    for c in constraints:
        a, b = c['coeffs']
        rhs = c['rhs']
        if a != 0: x_max = max(x_max, rhs/a if rhs!=0 else 10)
        if b != 0: y_max = max(y_max, rhs/b if rhs!=0 else 10)
        
    x_lim = x_max * 1.2 if x_max != 0 else 10
    y_lim = y_max * 1.2 if y_max != 0 else 10
    
    d = np.linspace(0, x_lim, 400)
    x, y = np.meshgrid(d, d)
    
    for i, c in enumerate(constraints):
        a, b = c['coeffs']
        rhs = c['rhs']
        type_ = c['type']
        
        x_vals = np.linspace(0, x_lim, 100)
        if b != 0:
            y_vals = (rhs - a*x_vals) / b
            ax.plot(x_vals, y_vals, label=f'{a}x1 + {b}x2 {type_} {rhs}')
        else:
            ax.axvline(x=rhs/a, label=f'{a}x1 {type_} {rhs}')
            
    feasible_grid = np.ones(x.shape, dtype=bool)
    
    for c in constraints:
        a, b = c['coeffs']
        rhs = c['rhs']
        type_ = c['type']
        
        if type_ == '<=': feasible_grid &= (a*x + b*y <= rhs)
        elif type_ == '>=': feasible_grid &= (a*x + b*y >= rhs)
        elif type_ == '=': feasible_grid &= (np.abs(a*x + b*y - rhs) < 0.1)
            
    feasible_grid &= (x >= 0) & (y >= 0)
    
    ax.imshow(feasible_grid.astype(int), extent=(0, x_lim, 0, x_lim), origin='lower', cmap='Greys', alpha=0.3)
    
    color = 'blue' if obj_type == 'Min' else 'red'
    ax.quiver(0, 0, obj_coeffs[0], obj_coeffs[1], color=color, scale=1, scale_units='xy', label='Obj Gradient')
    
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    ax.legend()
    ax.set_title("Feasible Region")
    
    return fig

# --- 4. Streamlit Interface ---
st.set_page_config(page_title="QTM Solver Pro", layout="wide")

st.title("📈 QTM Linear Programming Solver")
st.markdown("Solves **Simplex** & **Big M**, visualizes **2-variable** problems, formulates **Duals**, and calculates **Sensitivity**.")

with st.sidebar:
    st.header("Settings")
    method = st.radio("Method", ["Auto-Detect (Simplex/Big M)"])
    obj_type = st.radio("Objective", ["Max", "Min"])
    num_vars = st.number_input("Variables (x)", 1, 10, 2)
    num_const = st.number_input("Constraints", 1, 10, 2)

st.subheader("1. Define Problem")
st.markdown("#### Objective Function")
cols_obj = st.columns(num_vars)
obj_coeffs = []
for i in range(num_vars):
    val = cols_obj[i].number_input(f"Coeff x{i+1}", value=0.0, step=1.0, key=f"obj_{i}")
    obj_coeffs.append(val)

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
        val = cols[j].number_input(f"a{i+1}{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}", label_visibility="collapsed")
        row_coeffs.append(val)
    ctype = cols[num_vars].selectbox("Type", ["<=", ">=", "="], key=f"type_{i}", label_visibility="collapsed")
    rhs = cols[num_vars+1].number_input("RHS", value=0.0, step=1.0, key=f"rhs_{i}", label_visibility="collapsed")
    constraints.append({'coeffs': row_coeffs, 'type': ctype, 'rhs': rhs})

st.divider()

if st.button("Solve, Visualize & Analyze", type="primary"):
    
    # 1. Dual Formulation
    st.subheader("2. Dual Problem Formulation")
    dual_obj, dual_consts, dual_restr = formulate_dual(num_vars, constraints, obj_coeffs, obj_type)
    
    st.latex(dual_obj)
    st.markdown("**Subject to:**")
    for eq in dual_consts:
        st.latex(eq)
    st.markdown("**Where:**")
    for r in dual_restr:
        st.latex(r)
        
    st.divider()
    
    # 2. Plotting 
    if num_vars == 2:
        st.subheader("3. Graphical Visualization")
        fig = plot_lpp(constraints, obj_coeffs, obj_type)
        if fig: st.pyplot(fig)
        st.divider()
    
    # 3. Solver
    iter_header_num = 4 if num_vars == 2 else 3
    st.subheader(f"{iter_header_num}. Iterations (Tableau)")
    
    try:
        steps, status, z_val, vars_val, shadow_prices, reduced_costs = solve_linear_program(num_vars, constraints, obj_coeffs, obj_type)
        
        tabs = st.tabs([f"Iteration {i}" for i in range(len(steps))])
        for i, tab in enumerate(tabs):
            with tab:
                df = steps[i]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Highlighting the most negative Zj-Cj value (Entering Variable)
                st.dataframe(
                    df.style
                    .format(subset=numeric_cols, formatter="{:.2f}")
                    .highlight_min(axis=0, subset=pd.IndexSlice['Zj - Cj', numeric_cols[:-1]], color='#ffcccb')
                )
        
        st.divider()
        result_header_num = 5 if num_vars == 2 else 4
        st.subheader(f"{result_header_num}. Final Result & Sensitivity")
        
        if status == "Optimal":
            c1, c2 = st.columns(2)
            c1.success(f"**Optimal Z ({obj_type}) = {z_val:.4f}**")
            c2.write("**Decision Variables:**")
            c2.json(vars_val)
            
            # Sensitivity Analysis Display
            st.markdown("#### Sensitivity Analysis")
            s1, s2 = st.columns(2)
            
            with s1:
                st.info("**Shadow Prices (Resource Value)**\n\nShows how much $Z$ improves for each 1-unit increase in the constraint RHS.")
                st.json({k: round(v, 4) for k, v in shadow_prices.items()})
                
            with s2:
                st.info("**Reduced Costs (Opportunity Cost)**\n\nShows how much the cost coefficient must improve before a non-basic variable enters the solution.")
                st.json({k: round(v, 4) for k, v in reduced_costs.items()})
            
            # Export
            export_df = pd.DataFrame()
            for idx, step_df in enumerate(steps):
                temp = step_df.copy()
                temp['Iteration'] = idx
                export_df = pd.concat([export_df, temp])
            
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full Tableaus (CSV)", data=csv, file_name='qtm_solution.csv', mime='text/csv')
            
        else:
            st.error(f"Solver Status: {status}")
            
    except Exception as e:
        st.error(f"Error during calculation: {e}")
