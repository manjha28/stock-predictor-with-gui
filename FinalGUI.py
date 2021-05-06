import PySimpleGUI as sg

sg.ChangeLookAndFeel('DarkBlue')

# ------ Menu Definition ------ #
menu_def = [['&File', ['&Open', '&Save', 'E&xit', 'Properties']],
            ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
            ['&Help', '&About...'], ]

# column1 = [[sg.Text('Column 1', background_color='lightblue', justification='center', size=(10, 1))],
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
#            [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]

layout = [
    [sg.Menu(menu_def, tearoff=True)],
    [sg.Text('Trading System By Manish Jha!', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
    [sg.Frame('Services',[[sg.Button('Scanner'),sg.Button('Predictor'),sg.Button('MOving Avg Stg')]])]
]

window = sg.Window('Everything bagel', layout, default_element_size=(40, 1), grab_anywhere=False)
event, values = window.read()
window.close()

sg.Popup('Title',
         'The results of the window.',
         'The button clicked was "{}"'.format(event),
         'The values are', values)