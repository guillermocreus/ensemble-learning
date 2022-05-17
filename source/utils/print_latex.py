import numpy as np

def print_table(X, include_line_no=True):
    s = ""
    if len(X.shape) > 1:
        for ind, row in enumerate(X):
            s_row = ""
            if include_line_no:
                s_row += "\\textbf{\#" + str(ind + 1) + "} & & "
            s_row += " & ".join(np.round(row, 3).astype(str)) + "\\\\\n"
            s_row += "\hline\n"
            s += s_row
    else:
        if include_line_no:
            s += "\\textbf{\#1} & & "
        s += " & ".join(np.round(X, 3).astype(str)) + "\\\\\n"
    print(s)