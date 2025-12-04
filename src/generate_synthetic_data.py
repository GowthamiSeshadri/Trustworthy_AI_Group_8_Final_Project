import pandas as pd
import numpy as np

def generate_synthetic_hr_data(n=10000, seed=42):
    np.random.seed(seed)
    gender = np.random.choice(['Male', 'Female'], size=n)
    age = np.random.randint(21, 60, size=n)
    experience = np.random.randint(0, 15, size=n)
    education = np.random.choice(['Bachelors', 'Masters', 'PhD'], size=n)
    score = np.random.normal(70, 10, size=n)

    # Hidden bias: females get slightly lower hire probability
    hire_prob = 0.4 + 0.02 * experience + 0.01 * (score - 70)
    hire_prob -= (gender == 'Female') * 0.05
    hired = np.random.binomial(1, 1 / (1 + np.exp(-hire_prob)))

    df = pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Experience': experience,
        'Education': education,
        'Score': score,
        'Hired': hired
    })
    df.to_csv('data/hr_synthetic.csv', index=False)
    print('✅ Synthetic HR dataset saved to data/hr_synthetic.csv')
    return df

if __name__ == '__main__':
    df = generate_synthetic_hr_data()
    print(df.head())
