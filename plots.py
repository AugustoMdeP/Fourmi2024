import matplotlib.pyplot as plt
import numpy as np

# Defina os seus vetores de dimensão 2
vetores = [
    # (1, 5.86),  # Vetor 1
    # (1, 14.67),  # Vetor 2
    # (2, 6.52),  # Vetor 1
    # (2, 15.02),   # Vetor 2
    # (3, 6.04),  # Vetor 1
    # (3, 14.16),   # Vetor 2
    # (4, 5.69),  # Vetor 1
    # (4, 14.01)   # Vetor 2

    # (1, 4.97),  # Vetor 1
    # (1, 10.90),  # Vetor 2
    # (2, 5.00),  # Vetor 1
    # (2, 11.00),   # Vetor 2
    # (3, 5.04),  # Vetor 1
    # (3, 11.12),   # Vetor 2
    # (4, 4.99),  # Vetor 1
    # (4, 11.36)   # Vetor 2

    # (1, 5.32),  # Vetor 1
    # (1, 11.99),  # Vetor 2
    # (2, 6.52),  # Vetor 1
    # (2, 12.28),   # Vetor 2
    # (3, 5.20),  # Vetor 1
    # (3, 12.57),   # Vetor 2
    # (4, 5.05),  # Vetor 1
    # (4, 11.75)   # Vetor 2

    (1, 3.36),  # Vetor 1
    (1, 22.65),  # Vetor 2
    (2, 3.49),  # Vetor 1
    (2, 23.92),   # Vetor 2
    (3, 3.10),  # Vetor 1
    (3, 23.20),   # Vetor 2
    (4, 3.33),  # Vetor 1
    (4, 26.04)   # Vetor 2
]
# Cores para cada ponto
cores = ['r' if i % 2 != 0 else 'b' for i in range(len(vetores))]

# Separar os pontos vermelhos e azuis
pontos_vermelhos = [p[1] for p, cor in zip(vetores, cores) if cor == 'r']
pontos_azuis = [p[1] for p, cor in zip(vetores, cores) if cor == 'b']

# Calcular as médias dos pontos vermelhos e azuis
media_vermelhos = np.mean(pontos_vermelhos)
media_azuis = np.mean(pontos_azuis)

# Plotagem dos pontos
for i, ponto in enumerate(vetores):
    x, y = ponto  # Extrai as coordenadas x e y do ponto
    plt.scatter(x, y, color=cores[i], label=f'Ponto {i+1}', s=100)

# Adicionar as médias como texto no gráfico
plt.text(0, media_vermelhos - 2, f'Moyenne 500 nourriture: {media_vermelhos:.2f}s', color='r', verticalalignment='center')
plt.text(0, media_azuis +2, f'Moyenne 1 nourriture: {media_azuis:.2f}s', color='b', verticalalignment='center')

plt.xlabel('Essai')  # Rótulo do eixo x
plt.ylabel('Temps (s)')  # Rótulo do eixo y
plt.title('Performance deuxième parallélisation avec np=2')  # Título do gráfico
# plt.grid(True)  # Adiciona a grade
plt.xticks(np.arange(1, max([p[0] for p in vetores]) + 1, 1))  # Define os ticks do eixo x como inteiros
plt.yticks(np.arange(1, max([p[1] for p in vetores]) + 1, 1))  # Define os ticks do eixo y como inteiros
plt.axis('equal')  # Garante que os eixos tenham a mesma escala
plt.show()  # Mostra o gráfico
