# import json
#
# # Opening JSON file
# with open('a:\\202303_experiments_4\\results_50.json', ) as f:
#
#     # returns JSON object as
#     # a dictionary
#     data = json.load(f)
#
#     # Iterating through the json
#     # list
#     for i in data['emp_details']:
#         print(i)


import tkinter as tk

def gcd():
    a = int(entry1.get())
    b = int(entry2.get())
    while b != 0:
        a, b = b, a % b
    result_label.config(text="GCD: {}".format(a))

root = tk.Tk()
root.title("GCD Calculator")

label1 = tk.Label(root, text="Number 1:")
label1.grid(row=0, column=0)

entry1 = tk.Entry(root)
entry1.grid(row=0, column=1)

label2 = tk.Label(root, text="Number 2:")
label2.grid(row=1, column=0)

entry2 = tk.Entry(root)
entry2.grid(row=1, column=1)

calculate_button = tk.Button(root, text="Calculate GCD", command=gcd)
calculate_button.grid(row=2, column=0, columnspan=2)

result_label = tk.Label(root, text="")
result_label.grid(row=3, column=0, columnspan=2)

root.mainloop()
