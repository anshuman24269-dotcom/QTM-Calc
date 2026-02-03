import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Backend: Solver Logic (Big M Method) ---

def solve_linear_program(num_vars, constraints, objective_coeffs, obj_type='Max'):
    """
    Solves LPP using Big M Method. 
    Returns: steps (list of dataframes), status (str), final_z (float), final_vars (dict)
    """
    
    # --- Step A: Standardization ---
    # Convert Minimize to Maximize by multiplying objective by -1
    if obj_type == 'Min':
        objective_coeffs = [-1 * c for c in objective_coeffs]
    
    # Identify required variables
    # We have: Decision Vars + Slack (for <=) + Surplus (for >=) + Artificial (for >=, =)
    
    num_constraints = len(constraints)
    M = 10000.0  # Big M value for calculation
    
    # Structures to hold the tableau
    # Columns: [x1..xn, s1..sk, A1..Am, Sol]
    
    # We need to construct the header and matrix dynamically
    col_names = [f'x{i+1}' for i in range(num_vars)]
    
    # Coefficients in matrix
    matrix = [] 
    basic_vars_idx = [] # Indices of variables currently in basis
    cost_basis = [] # Cb values
    
    # Extended Objective Function (Cj)
    # Starts with decision vars
    cj = list(objective_coeffs)
    
    # Counters for extra vars
    slack_count = 0
    artificial_count = 0
    
    # Process constraints to build Matrix and Cj
    # We iterate twice: First to define columns (Slack/Artif), Second to fill matrix
    
    # Let's handle row by row
    temp_rows = []
    
    # Dictionary to map row index to its constraint type for later
    constraint_types = [c['type'] for c in constraints]
    
    # 1. Add Slacks/Surplus
    for i, c_data in enumerate(constraints):
        if c_data['type'] == '<=':
            col_names.append(f's{slack_count+1}')
            cj.append(0)
            slack_count += 1
        elif c_data['type'] == '>=':
            col_names.append(f's{slack_count+1}')
            cj.append(0) # Surplus has 0 cost
            slack_count += 1
    
    # 2. Add Artificials
    for i, c_data in enumerate(constraints):
        if c_data['type'] in ['>=', '=']:
            col_names.append(f'A{artificial_count+1}')
            cj.append(-M) # Big M penalty
            artificial_count += 1

    # Total columns (excluding 'Solution')
    total_cols = len(cj)
    
    # 3. Build the Tableau Rows
    curr_slack = 0
    curr_artif = 0
    
    for i, c_data in enumerate(constraints):
        row = list(c_data['coeffs']) # [x1, x2...]
        
        # Pad slacks/surplus
        # We need to place 1, -1, or 0 in the correct slack columns
        slack_part = [0] * slack_count
        artif_part = [0] * artificial_count
        
        rhs = c_data['rhs']
        
        if c_data['type'] == '<=':
            # Add Slack (+1)
            # The slack index for this constraint is based on order of appearance
            # We must track which slack corresponds to which row.
            # Simplified: we iterate types again
            pass 

    # --- SIMPLIFIED MATRIX BUILDER ---
    # Re-looping cleanly to build the full row with correct 0s and 1s
    
    slack_ptr = 0
    artif_ptr = 0
    
    for i, c_data in enumerate(constraints):
        row = list(c_data['coeffs']) # Decision vars
        
        # Slacks/Surplus Section
        slack_part = [0] * slack_count
        if c_data['type'] == '<=':
            slack_part[slack_ptr] = 1
            basic_vars_idx.append(num_vars + slack_ptr) # Basis is this slack
            slack_ptr += 1
        elif c_data['type'] == '>=':
            slack_part[slack_ptr] = -1
            # Basis will be the Artificial variable, not surplus
            slack_ptr += 1
        
        row.extend(slack_part)
        
        # Artificial Section
        artif_part = [0] * artificial_count
        if c_data['type'] in ['>=', '=']:
            artif_part[artif_ptr] = 1
            # Basis is this Artificial
            # Calculate index: num_vars + num_slacks + current_artif_ptr
            basic_vars_idx.append(num_vars + slack_count + artif_ptr)
            artif_ptr += 1
        elif c_data['type'] == '=':
            pass # Already handled artif
            
        row.extend(artif_part)
        row.append(c_data['rhs']) # Add Solution
        
        matrix.append(row)

    # Convert to Numpy
    tableau = np.array(matrix, dtype=float)
    cj = np.array(cj, dtype=float)
    
    steps = []
    
    # --- Iteration Loop ---
    max_iter = 20
    for it in range(max_iter):
        
        # Identify Cb (Cost of Basic Vars)
        cb = cj[basic_vars_idx]
        
        # Calculate Zj (Cb * Column)
        # Tableau excluding solution col
        body = tableau[:, :-1]
        zj = np.dot(cb, body)
        
        # Calculate Cj - Zj
        net_eval = cj - zj
        
        # Create Display DataFrame
        df = pd.DataFrame(tableau, columns=col_names + ['Sol'])
        df.insert(0, 'Basic Var', [col_names[i] for i in basic_vars_idx])
        
        # Add Cj-Zj row
        row_net = list(net_eval) + [np.nan]
        df.loc['Cj - Zj'] = [''] + row_net
        
        steps.append(df)
        
        # Optimality Check (For Maximize: all Cj-Zj <= 0)
        # Floating point tolerance
        if np.all(net_eval <= 1e-5):
            status = "Optimal"
            break
            
        # Entering Variable (Most Positive Cj-Zj)
        entering_col = np.argmax(net_eval)
        
        # Unbounded Check
        if np.all(tableau[:, entering_col] <= 0):
            status = "Unbounded"
            return steps, status, 0, {}
            
        # Ratio Test (Sol / Entering)
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
        
        # Update Basis Index
        basic_vars_idx[leaving_row] = entering_col
        
        # 1. Normalize Pivot Row
        tableau[leaving_row, :] /= pivot_val
        
        # 2. Gaussian Elimination for other rows
        for i in range(num_constraints):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i, :] -= factor * tableau[leaving_row, :]
    
    else:
        status = "Max Iterations Reached"

    # Extract Results
    final_z = np.dot(cj[basic_vars_idx], tableau[:, -1])
    if obj_type == 'Min':
        final_z *= -1 # Revert sign
        
    final_vars = {}
    for i, var_idx in enumerate(basic_vars_idx):
        if var_idx < num_vars: # Only care about decision vars
            final_vars[col_names[var_idx]] = tableau[i, -1]
            
    # Fill missing vars with 0
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
    
    # Grid range
    x_max = 0
    y_max = 0
    
    # Determine plotting range based on intercepts
    for c in constraints:
        # ax + by = rhs
        a, b = c['coeffs']
        rhs = c['rhs']
        
        if a != 0: x_max = max(x_max, rhs/a if rhs!=0 else 10)
        if b != 0: y_max = max(y_max, rhs/b if rhs!=0 else 10)
        
    x_lim = x_max * 1.2 if x_max != 0 else 10
    y_lim = y_max * 1.2 if y_max != 0 else 10
    
    # Create grid
    d = np.linspace(0, x_lim, 400)
    x, y = np.meshgrid(d, d)
    
    # Plot constraints
    # We will shade the INVALID regions and leave the feasible region white (or vice versa)
    # A cleaner way in matplotlib is to fill the polygon.
    
    # Let's plot lines first
    for i, c in enumerate(constraints):
        a, b = c['coeffs']
        rhs = c['rhs']
        type_ = c['type']
        
        # Line eq: ax + by = rhs => y = (rhs - ax)/b
        x_vals = np.linspace(0, x_lim, 100)
        if b != 0:
            y_vals = (rhs - a*x_vals) / b
            # Filter negative y (first quadrant)
            # y_vals[y_vals < 0] = np.nan
            ax.plot(x_vals, y_vals, label=f'{a}x1 + {b}x2 {type_} {rhs}')
        else:
            ax.axvline(x=rhs/a, label=f'{a}x1 {type_} {rhs}')
            
    # Shade Feasible Region
    # We check a grid of points
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
            # Equality is hard to shade on grid, usually just a line
            # We add a small tolerance for visual display
            feasible_grid &= (np.abs(a*x + b*y - rhs) < 0.1)
            
    # Non-negativity
    feasible_grid &= (x >= 0) & (y >= 0)
    
    ax.imshow(feasible_grid.astype(int), 
              extent=(0, x_lim, 0, x_lim), 
              origin='lower', cmap='Greys', alpha=0.3)
    
    # Plot Objective Vector (Gradient) roughly
    # Max Z = c1x + c2y
    # Normal vector is (c1, c2)
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
st.markdown("Solves **Simplex**, **Big M**, and visualizes **2-variable** problems.")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    method = st.radio("Method", ["Auto-Detect (Simplex/Big M)", "Standard Simplex"])
    obj_type = st.radio("Objective", ["Max", "Min"])
    
    num_vars = st.number_input("Variables (x)", 1, 10, 2)
    num_const = st.number_input("Constraints", 1, 10, 2)
    
    st.info("Note: 'Auto-Detect' will apply Big M if >= or = constraints are present.")

# Input Form
st.subheader("1. Define Problem")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Objective Function")
    obj_coeffs = []
    for i in range(num_vars):
        val = st.number_input(f"C{i+1} (Coeff for x{i+1})", value=0.0, step=1.0)
        obj_coeffs.append(val)

with col2:
    st.markdown("#### Constraints")
    constraints = []
    for i in range(num_const):
        c1, c2, c3 = st.columns([3, 1, 1])
        coeffs = []
        with c1:
            # Creating a mini-grid for coefficients
            cols_v = st.columns(num_vars)
            for j in range(num_vars):
                val = cols_v[j].number_input(f"a{i+1}{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}")
                coeffs.append(val)
        with c2:
            ctype = st.selectbox("Type", ["<=", ">=", "="], key=f"type_{i}")
        with c3:
            rhs = st.number_input("RHS", value=0.0, step=1.0, key=f"rhs_{i}")
            
        constraints.append({'coeffs': coeffs, 'type': ctype, 'rhs': rhs})

if st.button("Solve & Visualize", type="primary"):
    st.divider()
    
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
                st.dataframe(steps[i].style.format("{:.2f}").highlight_max(axis=0, subset=pd.IndexSlice['Cj - Zj', :], color='#d1e7dd'))
                if i == 0:
                    st.caption("Initial Tableau constructed with Slacks/Artificial variables.")
        
        st.divider()
        st.subheader("4. Final Result")
        
        if status == "Optimal":
            c1, c2 = st.columns(2)
            c1.success(f"**Optimal Z ({obj_type}) = {z_val:.4f}**")
            c2.write("Decision Variables:")
            c2.json(vars_val)
        else:
            st.error(f"Solver Status: {status}")
            
    except Exception as e:
        st.error(f"Error during calculation: {e}")
        st.write("Tip: Ensure RHS values are non-negative.")