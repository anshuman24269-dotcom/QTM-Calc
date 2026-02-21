import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. Backend: Solver Logic ---
def solve_linear_program(num_vars_original, var_restrictions, constraints, obj_coeffs_orig, obj_type='Max'):
    
    # Handle Unrestricted Variables
    col_names = []
    objective_coeffs = []
    var_map = {} # Maps original var index to its split columns
    
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
    
    if obj_type == 'Min':
        objective_coeffs = [-1 * c for c in objective_coeffs]
    
    num_constraints = len(constraints)
    M = 10000.0  
    
    cj = list(objective_coeffs)
    slack_count = 0
    artificial_count = 0
    
    initial_basic_cols = [] # For B^-1 tracking (Range of Feasibility)
    
    # Setup Columns
    for c_data in constraints:
        if c_data['type'] == '<=':
            col_names.append(f's{slack_count+1}')
            cj.append(0)
            slack_count += 1
        elif c_data['type'] == '>=':
            col_names.append(f's{slack_count+1}')
            cj.append(0) 
            slack_count += 1
            
    for c_data in constraints:
        if c_data['type'] in ['>=', '=']:
            col_names.append(f'A{artificial_count+1}')
            cj.append(-M) 
            artificial_count += 1

    matrix = [] 
    basic_vars_idx = [] 
    
    slack_ptr = 0
    artif_ptr = 0
    
    # Build Matrix
    for i, c_data in enumerate(constraints):
        orig_row = list(c_data['coeffs'])
        row = []
        
        # Expand row for unrestricted vars
        for j in range(num_vars_original):
            if var_restrictions[j] == 'Unrestricted':
                row.extend([orig_row[j], -orig_row[j]])
            else:
                row.append(orig_row[j])
                
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
            
        row.extend(slack_part)
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
        
        zj = np.dot(cb, body)
        net_eval = zj - cj 
        
        df = pd.DataFrame(tableau, columns=col_names + ['Sol'])
        df.insert(0, 'Basic Var', [col_names[i] for i in basic_vars_idx])
        
        row_net = list(net_eval) + [np.nan]
        df.loc['Zj - Cj'] = [''] + row_net
        steps.append(df)
        
        if np.all(net_eval >= -1e-5):
            status = "Optimal"
            break
            
        entering_col = np.argmin(net_eval)
        
        if np.all(tableau[:, entering_col] <= 0):
            status = "Unbounded"
            break
            
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
             break

        pivot_val = tableau[leaving_row, entering_col]
        basic_vars_idx[leaving_row] = entering_col
        tableau[leaving_row, :] /= pivot_val
        
        for i in range(num_constraints):
            if i != leaving_row:
                factor = tableau[i, entering_col]
