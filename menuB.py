# Import the necessary packages
from consolemenu import *
from consolemenu.items import *
from pruebas5 import RedRBF

# Create the menu
menu = ConsoleMenu("Programa de Redes Neuronales de Base Radial", "Elige un dataset")

# Create some items

# MenuItem is the base class for all items, it doesn't do anything when selected
menu_item = MenuItem("Iris.csv")
menu_item2 = MenuItem("Bill_authentication.csv")

# Pedir número de neuronas en la capa oculta
neuronasO = FunctionItem("Número de neuronas en la capa oculta", input, ["Ingrese el número de neuronas en la capa oculta: "])
# Pedir tasa de aprendizaje
tasa_aprendizaje = FunctionItem("Tasa de aprendizaje", input, ["Ingrese la tasa de aprendizaje: "])
# Pedir número de épocas
epocas = FunctionItem("Número de épocas", input, ["Ingrese el número de épocas: "])
# Pedir función de activación en la capa oculta
f_capa_oculta = FunctionItem("Función de activación en la capa oculta", input, ["¿Qué función de activación desea utilizar en la capa oculta?\n1. Gausiana\n2. Multicuadrática\n3. Multicuadrática inversa\nIngrese la opción: "])
# Pedir función de activación en la capa de salida
f_capa_salida = FunctionItem("Función de activación en la capa de salida", input, ["¿Qué función de activación desea utilizar en la capa de salida?\n1. Sigmoide\n2. Softmax\n3. Tangente hiperbólica\nIngrese la opción: "])

rbf = RedRBF(
    dataset = "Iris.csv",
)

""" # A FunctionItem runs a Python function when selected
function_item = FunctionItem("Call a Python function", input, ["Enter an input"])

# A CommandItem runs a console command
command_item = CommandItem("Run a console command",  "touch hello.txt")

# A SelectionMenu constructs a menu from a list of strings
selection_menu = SelectionMenu(["item1", "item2", "item3"])

# A SubmenuItem lets you add a menu (the selection_menu above, for example)
# as a submenu of another menu
submenu_item = SubmenuItem("Submenu item", selection_menu, menu) """

# Once we're done creating them, we just add the items to the menu
menu.append_item(menu_item)
menu.append_item(menu_item2)
menu.append_item(neuronasO)
menu.append_item(tasa_aprendizaje)
menu.append_item(epocas)
menu.append_item(f_capa_oculta)
# menu.append_item(function_item)
# menu.append_item(command_item)
# menu.append_item(submenu_item)

# Finally, we call show to show the menu and allow the user to interact
menu.show()