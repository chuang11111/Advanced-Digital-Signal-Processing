import numpy as np
import matplotlib.pyplot as plt

N = 19
fs = 8000
delta = 0.0001
k = int((N - 1) / 2)
F0 = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.35, 0.39, 0.43, 0.47, 0.5]) 
E1 = 500
iteration = 0

while True:

    # STEP2
    A = np.zeros([k + 2, k + 2])
    hd = np.zeros(k + 2)
    for m in range(0, k + 2):
        for n in range(0, k + 1):
            A[m, n] = np.cos(2 * np.pi * n * F0[m])
        cutoff = 2200 / fs
        A[m, k + 1] = (-1)**m * 1/(0.6 if F0[m] < cutoff else 1)
        hd[m] = int(F0[m] >= cutoff)

    s = np.linalg.solve(A, hd)

    # STEP3 
    F = np.linspace(0, 0.5, 5001)
    R = np.zeros([F.shape[0]])
    hd = np.zeros([F.shape[0]])
    w = np.zeros([F.shape[0]])
    for f in range(0, F.shape[0]):

        # R[f]
        temp = 0
        for n in range(0, k + 1):
            temp += s[n] * np.cos(2 * np.pi * n * F[f])
        R[f] = temp

        # hd[f]
        cutoff = 2200 / fs
        hd[f] = int(F[f] >= cutoff)

        # w[f]
        cutoff2 = 2000 / fs
        cutoff3 = 2400 / fs
        w[f] = (0.6 if F[f] <= cutoff2 else (1 if F[f] >= cutoff3 else 0))

    err = (R - hd) * w

    # STEP4
    extreme_point = []
    extreme_value = []
    boundary_point = []
    boundary_value = []

    n = F.shape[0]

    
    extreme_indices = np.where((err[1:-1] > err[:-2]) & (err[1:-1] > err[2:]))[0] + 1
    extreme_indices = np.append(extreme_indices, np.where((err[1:-1] < err[:-2]) & (err[1:-1] < err[2:]))[0] + 1)
    boundary_indices = np.array([0, len(err) - 1])
    
    extreme_point = F[extreme_indices]
    extreme_value = err[extreme_indices]
    boundary_point = F[boundary_indices]
    boundary_value = err[boundary_indices]
    
    while len(extreme_value) < (k + 2) and len(boundary_value) > 0:
        max_boundary_index = np.argmax(np.abs(boundary_value))
        extreme_point = np.append(extreme_point, boundary_point[max_boundary_index])
        extreme_value = np.append(extreme_value, boundary_value[max_boundary_index])
        boundary_point = np.delete(boundary_point, max_boundary_index)
        boundary_value = np.delete(boundary_value, max_boundary_index)
    
    sorted_indices = np.argsort(extreme_point)
    extreme_point = extreme_point[sorted_indices]
    extreme_value = extreme_value[sorted_indices]

    (extreme_point, extreme_value) = (list(t) for t in zip(*sorted(zip(extreme_point, extreme_value))))

    F0 = np.array(extreme_point)

    max_err = np.max(np.abs(err))
    max_err_idx = np.argmax(np.abs(err))
    plt.figure()
    plt.plot(F, err)
    plt.scatter(F[max_err_idx], err[max_err_idx], c='r', marker='o', s=80, label=f'Maximal Error = {max_err:.5f}')
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Error')
    plt.title(f'Iteration {iteration}')
    plt.show()

    if iteration >= 0:
        print(f'Iteration {iteration} max_err is : {max_err:.5f}')

    # STEP5 
    E0 = max(abs(err))

    if abs(E1-E0) < delta:
        break
    else:
        E1 = E0
        iteration += 1


# STEP6 
err = (R - hd) * w
extreme_point = []
extreme_value = []
boundary_point = []
boundary_value = []

n = F.shape[0]

for i in range(n):
    curr_err = err[i]
    if i == 0:
        if curr_err > 0 and curr_err > err[i + 1]:
            boundary_point.append(F[i])
            boundary_value.append(abs(curr_err))
        elif curr_err < 0 and curr_err <= err[i + 1]:
            boundary_point.append(F[i])
            boundary_value.append(abs(curr_err))
    elif i == n - 1:
        if curr_err > err[i - 1] and curr_err > 0:
            boundary_point.append(F[i])
            boundary_value.append(abs(curr_err))
        elif curr_err < err[i - 1] and curr_err < 0:
            boundary_point.append(F[i])
            boundary_value.append(abs(curr_err))
    else:
        prev_err = err[i - 1]
        next_err = err[i + 1]
        if curr_err > prev_err and curr_err > next_err:
            extreme_point.append(F[i])
            extreme_value.append(curr_err)
        elif curr_err < prev_err and curr_err < next_err:
            extreme_point.append(F[i])
            extreme_value.append(curr_err)

while len(extreme_value) < (k + 2):
    if boundary_value:
        temp_value = max(boundary_value)
        index = boundary_value.index(temp_value)
        boundary_value.remove(temp_value)
        extreme_value.append(temp_value)
        temp_point = boundary_point.pop(index)
        extreme_point.append(temp_point)
    else:
        break

(extreme_point, extreme_value) = (list(t) for t in zip(*sorted(zip(extreme_point, extreme_value))))


max_err = np.max(np.abs(err))
max_err_idx = np.argmax(np.abs(err))
plt.figure()
plt.plot(F, err)
plt.scatter(F[max_err_idx], err[max_err_idx], c='r', marker='o', s=80, label=f'maximal error = {max_err:.5f}')
plt.legend()
plt.xlabel('Frequency')
plt.ylabel('Error')
plt.title(f'Iteration {iteration+1}')
plt.show()

print(f'Iteration {iteration+1} max_err is : {max_err:.5f}')


# frequency response
extreme_R = [R[F == f][0] for f in extreme_point]

plt.figure(figsize=(8, 6))
plt.plot(F, R, label='H(F)')
plt.plot(F, hd, label='Hd(F)')
plt.scatter(extreme_point, extreme_R, color='black', marker='o', label='Extremes')
for i, (f, r) in enumerate(zip(extreme_point, extreme_R)):
    plt.text(f, r, f'{f:.4f}', fontsize=10, ha='right', va='bottom', color='black')


plt.axvline(x=2000/fs, color='black', linestyle=':', label='Transition range')
plt.axvline(x=2400/fs, color='black', linestyle=':', label='Transition range')

plt.xticks(np.arange(0, 0.55, 0.05))
plt.xlabel('Frequency')
plt.ylabel('Frequency Response')
plt.xlim(0, 0.5)
plt.ylim(-0.3, 1.25)
plt.legend()
plt.title('Frequency Response')
plt.grid(True)
plt.show()

# impulse response
h_n = np.zeros([N])
y = np.zeros([N])

for i in range(0, N):
    if i < k:
        h_n[i] = s[k-i] / 2
    elif i == k:
        h_n[i] = s[0]
    elif i > k:
        h_n[i] = s[i-k] / 2

n = np.arange(N)
plt.figure(figsize=(8, 6))
plt.stem(n, h_n)

plt.plot(n, y, color='black', linestyle='-', label='Zero Reference')
plt.xlabel('n', fontsize=14)
plt.ylabel('h[n]', fontsize=14)
plt.title('Impulse Response', fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
