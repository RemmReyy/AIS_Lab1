import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Створюємо вхідні змінні
x = ctrl.Antecedent(np.linspace(-1, 1, 100), 'x')
y = ctrl.Antecedent(np.linspace(-1, 1, 100), 'y')
z = ctrl.Consequent(np.linspace(-1, 1, 100), 'z')

# Визначаємо функції належності для входів
x['negative'] = fuzz.trimf(x.universe, [-1, -1, 0])
x['zero'] = fuzz.trimf(x.universe, [-0.5, 0, 0.5])
x['positive'] = fuzz.trimf(x.universe, [0, 1, 1])

y['negative'] = fuzz.trimf(y.universe, [-1, -1, 0])
y['zero'] = fuzz.trimf(y.universe, [-0.5, 0, 0.5])
y['positive'] = fuzz.trimf(y.universe, [0, 1, 1])

# Визначаємо функції належності для виходу
z['negative'] = fuzz.trimf(z.universe, [-1, -1, 0])
z['zero'] = fuzz.trimf(z.universe, [-0.5, 0, 0.5])
z['positive'] = fuzz.trimf(z.universe, [0, 1, 1])

# Створюємо правила
rule1 = ctrl.Rule(x['negative'] & y['negative'], z['positive'])
rule2 = ctrl.Rule(x['negative'] & y['zero'], z['zero'])
rule3 = ctrl.Rule(x['negative'] & y['positive'], z['negative'])
rule4 = ctrl.Rule(x['zero'] & y['negative'], z['zero'])
rule5 = ctrl.Rule(x['zero'] & y['zero'], z['zero'])
rule6 = ctrl.Rule(x['zero'] & y['positive'], z['zero'])
rule7 = ctrl.Rule(x['positive'] & y['negative'], z['negative'])
rule8 = ctrl.Rule(x['positive'] & y['zero'], z['zero'])
rule9 = ctrl.Rule(x['positive'] & y['positive'], z['positive'])

# Створюємо систему керування
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
simulation = ctrl.ControlSystemSimulation(system)

# Модифікована функція для обчислення реального значення
def real_function(x, y):
    under_sqrt = x**2 + y**3
    under_sqrt[under_sqrt < 0] = 0
    return 1 - np.abs(np.sqrt(under_sqrt))

# Створюємо сітку значень для візуалізації
x_range = np.linspace(-1, 1, 30)
y_range = np.linspace(-1, 1, 30)
X, Y = np.meshgrid(x_range, y_range)
Z_fuzzy = np.zeros_like(X)
Z_real = real_function(X, Y)

# Обчислюємо значення нечіткої системи для кожної точки
for i in range(len(x_range)):
    for j in range(len(y_range)):
        simulation.input['x'] = x_range[i]
        simulation.input['y'] = y_range[j]
        simulation.compute()
        Z_fuzzy[j, i] = simulation.output['z']

# Створюємо фігуру з трьома 3D підграфіками
fig = plt.figure(figsize=(20, 6))

# Реальна функція
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_real, cmap='viridis')
ax1.set_title('Реальна функція')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Нечітка система
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_fuzzy, cmap='viridis')
ax2.set_title('Нечітка система')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Різниця (похибка)
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, np.abs(Z_real - Z_fuzzy), cmap='viridis')
ax3.set_title('Абсолютна похибка')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# Налаштування відображення
for ax in [ax1, ax2, ax3]:
    ax.view_init(elev=30, azim=45)  # Кут огляду
    ax.set_zlim(-1, 1)  # Встановлюємо межі по осі z

plt.tight_layout()
plt.show()

# Виведення статистики похибок
error = np.abs(Z_real - Z_fuzzy)
print(f"Середня абсолютна похибка: {np.mean(error):.4f}")
print(f"Максимальна абсолютна похибка: {np.max(error):.4f}")
print(f"Мінімальна абсолютна похибка: {np.min(error):.4f}")