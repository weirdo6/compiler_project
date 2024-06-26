
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'ASSIGN BEGIN COMMA DIVIDE DOT END IDENTIFIER LPAREN MINUS NUMBER PLUS POW PROGRAM REAL RPAREN SEMICOLON TIMESprogram : PROGRAM IDENTIFIER SEMICOLON block DOTblock : BEGIN statement_list ENDstatement_list : statement\n                      | statement_list COMMA statementstatement : IDENTIFIER ASSIGN expressionexpression : expression PLUS term\n                  | expression MINUS termexpression : termterm : term TIMES factor\n            | term DIVIDE factorterm : factorfactor : factor POW factorfactor : LPAREN expression RPARENfactor : NUMBERfactor : REALfactor : IDENTIFIER'
    
_lr_action_items = {'PROGRAM':([0,],[2,]),'$end':([1,7,],[0,-1,]),'IDENTIFIER':([2,6,12,13,19,22,23,24,25,26,],[3,10,10,15,15,15,15,15,15,15,]),'SEMICOLON':([3,],[4,]),'BEGIN':([4,],[6,]),'DOT':([5,11,],[7,-2,]),'END':([8,9,14,15,16,17,18,20,21,28,29,30,31,32,33,],[11,-3,-4,-16,-5,-8,-11,-14,-15,-6,-7,-9,-10,-12,-13,]),'COMMA':([8,9,14,15,16,17,18,20,21,28,29,30,31,32,33,],[12,-3,-4,-16,-5,-8,-11,-14,-15,-6,-7,-9,-10,-12,-13,]),'ASSIGN':([10,],[13,]),'LPAREN':([13,19,22,23,24,25,26,],[19,19,19,19,19,19,19,]),'NUMBER':([13,19,22,23,24,25,26,],[20,20,20,20,20,20,20,]),'REAL':([13,19,22,23,24,25,26,],[21,21,21,21,21,21,21,]),'POW':([15,18,20,21,30,31,32,33,],[-16,26,-14,-15,26,26,26,-13,]),'TIMES':([15,17,18,20,21,28,29,30,31,32,33,],[-16,24,-11,-14,-15,24,24,-9,-10,-12,-13,]),'DIVIDE':([15,17,18,20,21,28,29,30,31,32,33,],[-16,25,-11,-14,-15,25,25,-9,-10,-12,-13,]),'PLUS':([15,16,17,18,20,21,27,28,29,30,31,32,33,],[-16,22,-8,-11,-14,-15,22,-6,-7,-9,-10,-12,-13,]),'MINUS':([15,16,17,18,20,21,27,28,29,30,31,32,33,],[-16,23,-8,-11,-14,-15,23,-6,-7,-9,-10,-12,-13,]),'RPAREN':([15,17,18,20,21,27,28,29,30,31,32,33,],[-16,-8,-11,-14,-15,33,-6,-7,-9,-10,-12,-13,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'program':([0,],[1,]),'block':([4,],[5,]),'statement_list':([6,],[8,]),'statement':([6,12,],[9,14,]),'expression':([13,19,],[16,27,]),'term':([13,19,22,23,],[17,17,28,29,]),'factor':([13,19,22,23,24,25,26,],[18,18,18,18,30,31,32,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> program","S'",1,None,None,None),
  ('program -> PROGRAM IDENTIFIER SEMICOLON block DOT','program',5,'p_program','compiler_experiment.py',101),
  ('block -> BEGIN statement_list END','block',3,'p_block','compiler_experiment.py',107),
  ('statement_list -> statement','statement_list',1,'p_statement_list','compiler_experiment.py',113),
  ('statement_list -> statement_list COMMA statement','statement_list',3,'p_statement_list','compiler_experiment.py',114),
  ('statement -> IDENTIFIER ASSIGN expression','statement',3,'p_statement','compiler_experiment.py',124),
  ('expression -> expression PLUS term','expression',3,'p_expression_binop','compiler_experiment.py',133),
  ('expression -> expression MINUS term','expression',3,'p_expression_binop','compiler_experiment.py',134),
  ('expression -> term','expression',1,'p_expression_term','compiler_experiment.py',139),
  ('term -> term TIMES factor','term',3,'p_term_binop','compiler_experiment.py',145),
  ('term -> term DIVIDE factor','term',3,'p_term_binop','compiler_experiment.py',146),
  ('term -> factor','term',1,'p_term_factor','compiler_experiment.py',151),
  ('factor -> factor POW factor','factor',3,'p_factor_pow','compiler_experiment.py',157),
  ('factor -> LPAREN expression RPAREN','factor',3,'p_factor_group','compiler_experiment.py',163),
  ('factor -> NUMBER','factor',1,'p_factor_number','compiler_experiment.py',169),
  ('factor -> REAL','factor',1,'p_factor_real','compiler_experiment.py',175),
  ('factor -> IDENTIFIER','factor',1,'p_factor_identifier','compiler_experiment.py',181),
]
