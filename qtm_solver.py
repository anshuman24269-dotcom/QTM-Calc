import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. Backend: Solver Logic (Big M Method) ---

def solve_linear_program(num_vars, constraints, objective_coeffs, obj_type='Max'):
    """
    Solves LPP using Big M Method. 
    Returns: steps (list of dataframes), status (str), final_z (float), final_vars (dict)
    """
    
    # --- Step A: Standardization ---
    if obj_type == 'Min':
        objective_coeffs = [-1 * c for c in objective_coeffs]
    
    num_constraints = len(constraints)
    M = 10000.0  # Big M value
    
    col_names = [f'x{i+1}' for i in range(num_vars)]
    
    cj = list(objective_coeffs)
    
    slack_count = 0
    artificial_count = 0
    
    # 1. Add Slacks/Surplus Headers
    for c_data in constraints:
        if c_data['type'] == '<=':
            col_names.append(f's{slack_count+1}')
            cj.append(0)
            slack_count += 1
        elif c_data['type'] == '>=':
            col_names.append(f's{slack_count+1}')
            cj.append(0) 
            slack_count += 1
    
    # 2. Add Artificials Headers
    for c_data in constraints:
        if c_data['type'] in ['>=', '=']:
            col_names.append(f'A{artificial_count+1}')
            cj.append(-M) 
            artificial_count += 1

    # 3. Build the Tableau Rows
    matrix = [] 
    basic_vars_idx = [] 
    
    slack_ptr = 0
    artif_ptr = 0
    
    for i, c_data in enumerate(constraints):
        row = list(c_data['coeffs']) 
        
        # Slack/Surplus Part
        slack_part = [0] * slack_count
        if c_data['type'] == '<=':
            slack_part[slack_ptr] = 1
            basic_vars_idx.append(num_vars + slack_ptr) 
            slack_ptr += 1
        elif c_data['type'] == '>=':
            slack_part[slack_ptr] = -1
            slack_ptr += 1
        
        row.extend(slack_part)
        
        # Artificial Part
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
    
    # --- Iteration Loop ---
    max_iter = 20
    status = "In Progress"
    
    for it in range(max_iter):
        
        # Calculate Zj and Cj - Zj
        cb = cj[basic_vars_idx]
        body = tableau[:, :-1]
        zj = np.dot(cb, body)
        net_eval = cj - zj
        
        # Create Display DataFrame
        df = pd.DataFrame(tableau, columns=col_names + ['Sol'])
        df.insert(0, 'Basic Var', [col_names[i] for i in basic_vars_idx])
        
        # Add Cj-Zj row
        row_net = list(net_eval) + [np.nan]
        df.loc['Cj - Zj'] = [''] + row_net
        
        steps.append(df)
        
        # Optimality Check
        if np.all(net_eval <= 1e-5):
            status = "Optimal"
            break
            
        # Entering Variable
        entering_col = np.argmax(net_eval)
        
        # Unbounded Check
        if np.all(tableau[:, entering_col] <= 0):
            status = "Unbounded"
            return steps, status, 0, {}
            
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
             return steps, status, 0, {}

        # Pivot
        pivot_val = tableau[leaving_row, entering_col]
        basic_vars_idx[leaving_row] = entering_col
        
        tableau[leaving_row, :] /= pivot_val
        
        for i in range(num_constraints):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i, :] -= factor * tableau[leaving_row, :]
    
    else:
        status = "Max Iterations Reached"

    # Extract Results
    final_z = np.dot(cj[basic_vars_idx], tableau[:, -1])
    if obj_type == 'Min':
        final_z *= -1 
        
    final_vars = {}
    for i, var_idx in enumerate(basic_vars_idx):
        if var_idx < num_vars: 
            final_vars[col_names[var_idx]] = tableau[i, -1]
            
    for i in range(num_vars):
        vname = f'x{i+1}'
        if vname not in final_vars:
            final_vars[vname] = 0.0
            
    return steps, status, final_z, final_vars

# --- 2. Visualization Logic ---

def plot_lpp(constraints, obj_coeffs, obj_type):
    """
    Plots the feasible region for 2-variable problems.
    """
    if len(obj_coeffs) != 2:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Grid range determination
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
    
    # Plot constraints lines
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
            
    # Shade Feasible Region
    feasible_grid = np.ones(x.shape, dtype=bool)
    
    for c in constraints:
        a, b = c['coeffs']
        rhs = c['rhs']
        type_ = c['type']
        
        if type_ == '<=':
            feasible_grid &= (a*x + b*y <= rhs)
        elif type_ == '>=':
            feasible_grid &= (a*x + b*y >= rhs)
        elif type_ == '=':
            feasible_grid &= (np.abs(a*x + b*y - rhs) < 0.1)
            
    feasible_grid &= (x >= 0) & (y >= 0)
    
    ax.imshow(feasible_grid.astype(int), 
              extent=(0, x_lim, 0, x_lim), 
              origin='lower', cmap='Greys', alpha=0.3)
    
    # Objective Direction
    if obj_type == 'Min':
        # For Min, gradient points opposite to improvement, but usually we show gradient
        ax.quiver(0, 0, obj_coeffs[0], obj_coeffs[1], color='blue', scale=1, scale_units='xy', label='Obj Gradient')
    else:
        ax.quiver(0, 0, obj_coeffs[0], obj_coeffs[1], color='red', scale=1, scale_units='xy', label='Obj Direction')
    
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    ax.legend()
    ax.set_title("Feasible Region (Shaded)")
    
    return fig

# --- 3. Streamlit Interface ---

st.set_page_config(page_title="QTM Solver Pro", layout="wide")

st.title("📈 QTM Linear Programming Solver")
st.markdown("Solves **Simplex** & **Big M**, visualizes **2-variable** problems, and exports results.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    method = st.radio("Method", ["Auto-Detect (Simplex/Big M)"])
    obj_type = st.radio("Objective", ["Max", "Min"])
    
    num_vars = st.number_input("Variables (x)", 1, 10, 2)
    num_const = st.number_input("Constraints", 1, 10, 2)
    
    st.info("System automatically applies Big M if >= or = constraints are detected.")

# --- Improved Layout Section ---
st.subheader("1. Define Problem")

# Objective Function
st.markdown("#### Objective Function")
cols_obj = st.columns(num_vars)
obj_coeffs = []
for i in range(num_vars):
    val = cols_obj[i].number_input(f"Coeff x{i+1}", value=0.0, step=1.0, key=f"obj_{i}")
    obj_coeffs.append(val)

st.divider()

# Constraints (New Grid Layout)
st.markdown("#### Constraints")
constraints = []

# Headers for the grid
header_cols = st.columns(num_vars + 2)
for j in range(num_vars):
    header_cols[j].markdown(f"**Coeff x{j+1}**")
header_cols[num_vars].markdown("**Type**")
header_cols[num_vars+1].markdown("**RHS**")

for i in range(num_const):
    # Create a row of columns for each constraint
    cols = st.columns(num_vars + 2)
    row_coeffs = []
    
    # Variable Coefficients
    for j in range(num_vars):
        val = cols[j].number_input(f"a{i+1}{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}", label_visibility="collapsed")
        row_coeffs.append(val)
        
    # Type Selection
    ctype = cols[num_vars].selectbox("Type", ["<=", ">=", "="], key=f"type_{i}", label_visibility="collapsed")
    
    # RHS Value
    rhs = cols[num_vars+1].number_input("RHS", value=0.0, step=1.0, key=f"rhs_{i}", label_visibility="collapsed")
            
    constraints.append({'coeffs': row_coeffs, 'type': ctype, 'rhs': rhs})

st.divider()

if st.button("Solve & Visualize", type="primary"):
    
    # 1. Plotting (Only if 2 vars)
    if num_vars == 2:
        st.subheader("2. Graphical Visualization")
        fig = plot_lpp(constraints, obj_coeffs, obj_type)
        if fig:
            st.pyplot(fig)
        else:
            st.write("Could not generate plot.")
    
    # 2. Solver
    st.subheader("3. Iterations (Tableau)")
    
    try:
        steps, status, z_val, vars_val = solve_linear_program(num_vars, constraints, obj_coeffs, obj_type)
        
        # Display Steps
        tabs = st.tabs([f"Iteration {i}" for i in range(len(steps))])
        for i, tab in enumerate(tabs):
            with tab:
                # FIXED: Identify numeric columns for safe formatting
                df = steps[i]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                st.dataframe(
                    df.style
                    .format(subset=numeric_cols, formatter="{:.2f}")
                    .highlight_max(axis=0, subset=pd.IndexSlice['Cj - Zj', numeric_cols[:-1]], color='#d1e7dd')
                )
                
                if i == 0:
                    st.caption("Initial Tableau constructed with Slacks/Artificial variables.")
        
        st.divider()
        st.subheader("4. Final Result")
        
        if status == "Optimal":
            c1, c2 = st.columns(2)
            c1.success(f"**Optimal Z ({obj_type}) = {z_val:.4f}**")
            c2.write("Decision Variables:")
            c2.json(vars_val)
            
            # --- NEW: Export Functionality ---
            st.subheader("5. Export")
            
            # Combine all steps into one CSV
            export_df = pd.DataFrame()
            for idx, step_df in enumerate(steps):
                temp = step_df.copy()
                temp['Iteration'] = idx
                export_df = pd.concat([export_df, temp])
            
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Solution as CSV",
                data=csv,
                file_name='qtm_solution.csv',
                mime='text/csv',
            )
            
        else:
            st.error(f"Solver Status: {status}")
            
    except Exception as e:
        st.error(f"Error during calculation: {e}")
        st.write("Tip: Ensure RHS values are non-negative.")
