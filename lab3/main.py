import pandas as pd

N_COMPANIES = 10
data = pd.read_excel('data.xlsx', sheet_name='Лист1')

# Ініціалізація всіх показників
Kp, Kt, Kd, Kn = [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES
Pn, Pr, Pra, Pz, Pg = [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES
Cs, Ch, Co, Ct = [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES
Di, Dp, Dd = [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES
Rs, Rn, Rz = [0]*N_COMPANIES, [0]*N_COMPANIES, [0]*N_COMPANIES

# Заповнення даних
for k in range(N_COMPANIES):
    col = data[f'Компанія {k+1}']
    Kp[k], Kt[k], Kd[k], Kn[k] = col[0:4]
    Pn[k], Pr[k], Pra[k], Pz[k], Pg[k] = col[4:9]
    Cs[k], Ch[k], Co[k], Ct[k] = col[9:13]
    Di[k], Dp[k], Dd[k] = col[13:16]
    Rs[k], Rn[k], Rz[k] = col[16:19]

def koef_calculations(koef):
    total = sum(koef)
    return [k / total for k in koef]

def multicriteria_performance_evaluation():
    F1, F2, F3, F4, F5 = [[0] * N_COMPANIES for _ in range(5)]

    # Обчислення сум для нормалізації
    sums = {
        'Kp': sum(Kp), 'Kt': sum(Kt), 'Kd': sum(1/x for x in Kd), 'Kn': sum(1/x for x in Kn),
        'Pn': sum(1/x for x in Pn), 'Pr': sum(1/x for x in Pr), 'Pra': sum(1/x for x in Pra),
        'Pz': sum(1/x for x in Pz), 'Pg': sum(1/x for x in Pg),
        'Cs': sum(1/x for x in Cs), 'Ch': sum(1/x for x in Ch),
        'Co': sum(1/x for x in Co), 'Ct': sum(1/x for x in Ct),
        'Di': sum(Di), 'Dp': sum(1/x for x in Dp), 'Dd': sum(1/x for x in Dd),
        'Rs': sum(1/x for x in Rs), 'Rn': sum(1/x for x in Rn), 'Rz': sum(Rz)
    }

    koef = koef_calculations([1] * 19)

    for i in range(N_COMPANIES):
        # Нормалізація
        Kp0, Kt0 = Kp[i] / sums['Kp'], Kt[i] / sums['Kt']
        Kd0, Kn0 = (1 / Kd[i]) / sums['Kd'], (1 / Kn[i]) / sums['Kn']
        Pn0, Pr0 = (1 / Pn[i]) / sums['Pn'], (1 / Pr[i]) / sums['Pr']
        Pra0, Pz0, Pg0 = (1 / Pra[i]) / sums['Pra'], (1 / Pz[i]) / sums['Pz'], (1 / Pg[i]) / sums['Pg']
        Cs0, Ch0 = (1 / Cs[i]) / sums['Cs'], (1 / Ch[i]) / sums['Ch']
        Co0, Ct0 = (1 / Co[i]) / sums['Co'], (1 / Ct[i]) / sums['Ct']
        Di0, Dp0 = Di[i] / sums['Di'], (1 / Dp[i]) / sums['Dp']
        Dd0 = (1 / Dd[i]) / sums['Dd']
        Rs0, Rn0 = (1 / Rs[i]) / sums['Rs'], (1 / Rn[i]) / sums['Rn']
        Rz0 = Rz[i] / sums['Rz']

        # Адитивні критерії
        F1[i] = koef[0]/(1-Kp0) + koef[1]/(1-Kt0) + koef[2]/(1-Kd0) + koef[3]/(1-Kn0)
        F2[i] = koef[4]/(1-Pn0) + koef[5]/(1-Pr0) + koef[6]/(1-Pra0) + koef[7]/(1-Pz0) + koef[8]/(1-Pg0)
        F3[i] = koef[9]/(1-Cs0) + koef[10]/(1-Ch0) + koef[11]/(1-Co0) + koef[12]/(1-Ct0)
        F4[i] = koef[13]/(1-Di0) + koef[14]/(1-Dp0) + koef[15]/(1-Dd0)
        F5[i] = koef[16]/(1-Rs0) + koef[17]/(1-Rn0) + koef[18]/(1-Rz0)

    # Нормалізація F1...F5
    F10 = [f / (max(F1)+0.2) for f in F1]
    F20 = [f / (max(F2)+0.2) for f in F2]
    F30 = [f / (max(F3)+0.2) for f in F3]
    F40 = [f / (max(F4)+0.2) for f in F4]
    F50 = [f / (max(F5)+0.2) for f in F5]

    # Інтегральна оцінка
    k = koef_calculations([1]*5)
    I = [k[0]/(1-f1) + k[1]/(1-f2) + k[2]/(1-f3) + k[3]/(1-f4) + k[4]/(1-f5)
         for f1, f2, f3, f4, f5 in zip(F10, F20, F30, F40, F50)]

    maxI = k[0]/(1 - min(F1)) + k[1]/(1 - min(F2)) + k[2]/(1 - min(F3)) + k[3]/(1 - min(F4)) + k[4]/(1 - min(F5))
    I0 = [maxI / x for x in I]

    best_index = max(range(N_COMPANIES), key=lambda i: I0[i])

    print('Інтегральні оцінки:')
    for i, val in enumerate(I0):
        print(f'Компанія {i+1}: {val:.4f}')
    print('-'*50)
    print('Найкраща компанія:', best_index + 1)

    return best_index + 1

multicriteria_performance_evaluation()
