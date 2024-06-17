import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, Menu
import ply.lex as lex

# Description:
# This module defines the lexical rules for a simple expression parser.
# It supports basic arithmetic operations, control structures, and identifiers
# using Python's PLY library.

# List of token names. This is always required
tokens = (
    'NUMBER',       # Integer or decimal number
    'REAL',         # Floating point number
    'IDENTIFIER',   # Variable names
    'PLUS',         # Addition symbol
    'MINUS',        # Subtraction symbol
    'TIMES',        # Multiplication symbol
    'DIVIDE',       # Division symbol
    'POW',          # Exponentiation symbol
    'LPAREN',       # Left parenthesis
    'RPAREN',       # Right parenthesis
    'ASSIGN',       # Assignment operator
    'COMMA',        # Comma separator
    'BEGIN',        # Begin keyword for block start
    'END',          # End keyword for block end
    'SEMICOLON',    # Semicolon separator
    'PROGRAM',      # Program keyword
    'DOT'           # Dot for end of program
)

# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_POW = r'\^'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_ASSIGN = r'='
t_COMMA = r','
t_SEMICOLON = r';'
t_DOT = r'\.'
t_ignore = ' \t'  # Ignore spaces and tabs

# Function to handle REAL numbers
def t_REAL(t):
    r'\d+\.\d+'
    t.value = float(t.value)  # Convert string to float
    return t

# Function to handle integer numbers
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)  # Convert string to integer
    return t

# Function to handle identifiers and reserved keywords
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    # Check for reserved words and assign specific token types
    if t.value == 'begin':
        t.type = 'BEGIN'
    elif t.value == 'end':
        t.type = 'END'
    elif t.value == 'program':
        t.type = 'PROGRAM'
    return t

# Function to handle newlines and increment line number
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Error handling rule
def t_error(t):
    raise CompilerError(f"Illegal character '{t.value[0]}'", t.lineno, t.lexpos)
    t.lexer.skip(1)  # Skip the bad character (not reached due to exception)

# Usage:
# This module should be imported and used in conjunction with a parser module
# that can handle the tokens produced by these lexical rules. The lexer
# processes input text to produce tokens, which are then consumed by the parser
# to build an abstract syntax tree or perform other forms of syntactic analysis.


lexer = lex.lex()

import ply.yacc as yacc

# Description:
# This module defines the syntax rules and actions for a parser using Python's PLY library.
# It processes tokens from a lexer to construct an Abstract Syntax Tree (AST) and manages a symbol table.

# Global symbol table to store variable values during parsing
symbol_table = {}
# List to record shift-reduce actions for debugging or analysis
actions = []

# Production rules for the 'program' structure
def p_program(p):
    'program : PROGRAM IDENTIFIER SEMICOLON block DOT'
    p[0] = p[4]
    actions.append(f"Reduce by rule: program -> PROGRAM IDENTIFIER SEMICOLON block DOT")

# Production rules for 'block' structure, encapsulating a statement list
def p_block(p):
    'block : BEGIN statement_list END'
    p[0] = p[2]
    actions.append(f"Reduce by rule: block -> BEGIN statement_list END")

# Production rules for 'statement_list'
def p_statement_list(p):
    '''statement_list : statement
                      | statement_list COMMA statement'''
    if len(p) == 2:
        p[0] = [p[1]]
        actions.append(f"Reduce by rule: statement_list -> statement")
    else:
        p[0] = p[1] + [p[3]]
        actions.append(f"Reduce by rule: statement_list -> statement_list COMMA statement")

# Production rules for 'statement' dealing with assignments
def p_statement(p):
    'statement : IDENTIFIER ASSIGN expression'
    var_name = p[1]
    p[0] = ('assign', var_name, p[3])
    # Update or add variable to the symbol table
    # symbol_table[var_name] = evaluate_expression(p[3])
    actions.append(f"Reduce by rule: statement -> IDENTIFIER ASSIGN expression")

# Production rules for expressions involving binary operations
def p_expression_binop(p):
    '''expression : expression PLUS term
                  | expression MINUS term'''
    p[0] = ('binop', p[2], p[1], p[3])
    actions.append(f"Reduce by rule: expression -> expression {p[2]} term")

def p_expression_term(p):
    'expression : term'
    p[0] = p[1]
    actions.append("Reduce by rule: expression -> term")

# Production rules for terms involving multiplication or division
def p_term_binop(p):
    '''term : term TIMES factor
            | term DIVIDE factor'''
    p[0] = ('binop', p[2], p[1], p[3])
    actions.append(f"Reduce by rule: term -> term {p[2]} factor")

def p_term_factor(p):
    'term : factor'
    p[0] = p[1]
    actions.append("Reduce by rule: term -> factor")

# Handling power operations within factors
def p_factor_pow(p):
    'factor : factor POW factor'
    p[0] = ('binop', '^', p[1], p[3])
    actions.append(f"Reduce by rule: factor -> factor POW factor")

# Production rule for factors that are grouped expressions
def p_factor_group(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]
    actions.append("Reduce by rule: factor -> LPAREN expression RPAREN")

# Production rules for number literals
def p_factor_number(p):
    'factor : NUMBER'
    p[0] = ('number', p[1])
    actions.append("Reduce by rule: factor -> NUMBER")

# Production rules for real number literals
def p_factor_real(p):
    'factor : REAL'
    p[0] = ('real', p[1])
    actions.append("Reduce by rule: factor -> REAL")

# Production rules for identifiers used as factors
def p_factor_identifier(p):
    'factor : IDENTIFIER'
    var_name = p[1]
    # Ensure the variable is defined in the symbol table before using
    # if var_name not in symbol_table:
    #     raise Exception(f"Undefined variable '{var_name}' used in expression")
    p[0] = ('identifier', var_name)
    actions.append("Reduce by rule: factor -> IDENTIFIER")

# Error handling during parsing
def p_error(p):
    if p:
        raise CompilerError(f"Syntax error at token {p.type}, value '{p.value}'", p.lineno, p.lexpos)
    else:
        raise CompilerError("Syntax error at EOF")

# Function to evaluate expressions based on the constructed AST
def evaluate_expression(node):
    # Handles different types of nodes in AST
    if node[0] == 'number':
        return node[1]
    elif node[0] == 'real':
        return node[1]
    elif node[0] == 'identifier':
        if node[1] in symbol_table:
            return symbol_table[node[1]]
        else:
            raise Exception(f"Undefined variable '{node[1]}'")
    elif node[0] == 'binop':
        left = evaluate_expression(node[2])
        right = evaluate_expression(node[3])
        if node[1] == '+':
            return left + right
        elif node[1] == '-':
            return left - right
        elif node[1] == '*':
            return left * right
        elif node[1] == '/':
            return left / right
        elif node[1] == '^':
            return left ** right
    else:
        raise Exception(f"Invalid expression node '{node}'")



# Description:
# This module defines exception handling, program execution, AST formatting,
# and intermediate code generation for a simple programming language compiler.

class CompilerError(Exception):
    """
    Custom exception class for compiler errors that includes information about the error location.
    """
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.line = line
        self.column = column
        self.message = message

    def __str__(self):
        """
        Return a string representation of the error, including the line and column if available.
        """
        error_message = f"Error: {self.message}"
        if self.line is not None and self.column is not None:
            error_message += f" at line {self.line}, column {self.column}"
        return error_message

def execute_program(parser, lexer, input_program):
    """
    Executes the program by parsing the input and evaluating the expressions.
    :param parser: The parser instance
    :param lexer: The lexer instance
    :param input_program: A string containing the source code to be compiled
    :return: Tuple (bool, result) where bool indicates success, and result is the formatted AST or error message
    """
    try:
        ast = parser.parse(input_program, lexer=lexer)
        if ast is None:
            raise CompilerError("Parsing failed, AST was not generated.")
        symbol_table = {}  # Symbol table for storing variable values
        # for statement in ast:
        #     if statement[0] == 'assign':
        #         symbol_table[statement[1]] = evaluate_expression(statement[2])

        formatted_ast = format_ast(ast)  # Properly format the AST for output
        return True, formatted_ast

    except CompilerError as e:
        return False, f"Error: {e.message} at line {e.line}, position {e.column}"
    except Exception as e:
        return False, f"Unhandled exception: {str(e)}"

def format_ast(node, level=0):
    """
    Formats the AST for display.
    :param node: The root or any node of the AST
    :param level: Current depth in the AST, used for indentation
    :return: Formatted string representation of the AST
    """
    indent = "  " * level
    result = ""
    if isinstance(node, list):
        for child in node:
            result += format_ast(child, level)
    elif node[0] == 'assign':
        result += f"{indent}Assignment: {node[1]} =\n"
        result += format_ast(node[2], level + 1)
    elif node[0] == 'binop':
        result += f"{indent}BinaryOp: {node[1]}\n"
        result += format_ast(node[2], level + 1)
        result += format_ast(node[3], level + 1)
    elif node[0] == 'number':
        result += f"{indent}Number: {node[1]}\n"
    elif node[0] == 'real':
        result += f"{indent}Real: {node[1]}\n"
    elif node[0] == 'identifier':
        result += f"{indent}Identifier: {node[1]}\n"
    else:
        result += f"{indent}Unknown node: {node}\n"
    return result

def generate_intermediate_code(statements):
    """
    Generates intermediate code for a given set of statements.
    :param statements: List of statements from which to generate code
    :return: Formatted string of the intermediate code
    """
    code = []
    temp_counter = 1

    def get_temp():
        nonlocal temp_counter
        temp_name = f"T{temp_counter}"
        temp_counter += 1
        return temp_name

    def process_expression(expr):
        if isinstance(expr, tuple):
            if expr[0] == 'binop':
                op, left_expr, right_expr = expr[1], expr[2], expr[3]
                left_temp = process_expression(left_expr)
                right_temp = process_expression(right_expr)
                result_temp = get_temp()
                code.append((op, left_temp, right_temp, result_temp))
                return result_temp
            elif expr[0] in ('number', 'real'):
                return str(expr[1])
            elif expr[0] == 'identifier':
                return expr[1]
        elif isinstance(expr, str):  # Assuming it is a variable identifier
            return expr

    for statement in statements:
        if statement[0] == 'assign':
            identifier, expression = statement[1], statement[2]
            result_temp = process_expression(expression)
            code.append(('=', result_temp, '', identifier))

    # Format the code into a readable string with proper numbering using underscores
    formatted_code = "\n".join(f"①{i + 1} ({item[0]},_{item[1]},_{item[2]},_{item[3]})" for i, item in enumerate(code))
    return formatted_code

def get_tokens(input_program):
    try:
        lexer = lex.lex()
        lexer.input(input_program)
        tokens_info = ""
        while True:
            tok = lexer.token()
            if not tok:
                break  # No more tokens, end of input
            token_info = f"Type: {tok.type}, Value: {tok.value}, Line: {tok.lineno}, Position: {tok.lexpos}\n"
            tokens_info += token_info
        if not tokens_info:  # If no tokens were generated, check for only invalid inputs
            return False, "No valid tokens were generated."
        return True, tokens_info.strip()
    except CompilerError as e:
        return False, f"Error: {e.message} at line {e.line}, position {e.column}"
    except Exception as e:
        return False, str(e)

class CompilerApp:
    """
    A GUI application for a compiler interface using Tkinter.
    Provides features for file operations, lexical, syntax, and semantic analysis of code.
    """
    def __init__(self, master):
        """
        Initialize the application with a master window and set up the user interface.
        :param master: The master Tkinter window.
        """
        self.master = master
        self.ast = None  # To store the Abstract Syntax Tree
        master.title("Advanced Compiler Interface")

        # Set up the main menu bar
        menu_bar = Menu(master)
        master.config(menu=menu_bar)

        # Add commands directly to the menu bar for file operations and analyses
        menu_bar.add_command(label="Open File", command=self.open_file)
        menu_bar.add_command(label="Save File", command=self.save_file)
        menu_bar.add_command(label="Exit", command=self.exit_program, background='red')  # Highlight the Exit button

        # Add commands for analysis options
        menu_bar.add_command(label="Lexical Analysis", command=self.lexical_analysis)

        # Create a dropdown menu for syntax analysis options
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Syntax Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Show Shift-Reduce Actions", command=self.show_actions)
        analysis_menu.add_command(label="Show AST", command=self.show_ast)

        menu_bar.add_command(label="Semantic Analysis", command=self.semantic_analysis)
        menu_bar.add_command(label="Calculate", command=self.calculate)

        # Set up text areas for input and output
        self.input_text = scrolledtext.ScrolledText(master, height=10, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.output_text = scrolledtext.ScrolledText(master, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def open_file(self):
        """Opens a file dialog to select and read a file, loading its contents into the input text area."""
        file_path = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(tk.END, content)

    def save_file(self):
        """Opens a file dialog to select a path and save the contents of the input text area to a file."""
        file_path = filedialog.asksaveasfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'w') as file:
                content = self.input_text.get(1.0, tk.END)
                file.write(content)

    def exit_program(self):
        """Confirms with the user and exits the application."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.quit()

    def lexical_analysis(self):
        """Performs lexical analysis on the input text and displays the tokens in the output text area."""
        input_program = self.input_text.get(1.0, tk.END)
        is_correct, info = get_tokens(input_program)
        if not is_correct:
            messagebox.showerror("Error", info)
        else:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "\n" + info)

    def show_actions(self):
        """Displays the shift-reduce actions from the parser in the output text area."""
        self.output_text.delete(1.0, tk.END)
        formatted_actions = "\n".join(actions)
        self.output_text.insert(tk.END, formatted_actions)

    def show_ast(self):
        """Performs syntax analysis, checks the correctness of the program, and displays the AST if valid."""
        global symbol_table
        symbol_table = {}  # Reset symbol table before parsing

        parser = yacc.yacc()
        input_program = self.input_text.get(1.0, tk.END)
        lexer = lex.lex()

        is_correct, info = execute_program(parser, lexer, input_program)
        if not is_correct:
            messagebox.showerror("Syntax Error", info)
        else:
            self.ast = parser.parse(input_program, lexer=lexer)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Syntax Analysis: Program is correct\n" + info)

    def semantic_analysis(self):
        """Performs semantic analysis on the parsed AST and displays the generated intermediate code."""
        input_program = self.input_text.get(1.0, tk.END)
        try :
            for statement in self.ast:
                if statement[0] == 'assign':
                    symbol_table[statement[1]] = evaluate_expression(statement[2])
            intermediate_code = generate_intermediate_code(self.ast)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, intermediate_code)
        except Exception as e:
            messagebox.showerror("semantic_analysis Error", e)



    def calculate(self):
        """Calculates the expressions in the input program using the symbol table and displays the results."""
        result = symbol_table
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)



if __name__ == "__main__":
    root = tk.Tk()
    app = CompilerApp(root)
    root.mainloop()
