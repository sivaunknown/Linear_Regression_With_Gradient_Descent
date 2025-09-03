import matplotlib.pyplot as plt
import pandas as pd

class SLR_Gradient_Descent():

    def __init__(self, data : pd.DataFrame,learning_rate : float):
        
        self.data : pd.DataFrame = data
        self.learning_rate : float = learning_rate

    def gradient_desc(self,m_now : float , b_now : float) -> float:

        m_gradient : float = 0
        b_gradient : float = 0
        n : int = len(self.data)

        for i in range(n):
            x = self.data.iloc[i].YearsExperience
            y = self.data.iloc[i].Salary
            m_gradient  += -(2/n) * x * (y - (m_now * x + b_now))
            b_gradient  += -(2/n) * (y - (m_now * x + b_now))
            
        m_gradient = m_now - self.learning_rate * m_gradient
        b_gradient = b_now - self.learning_rate * b_gradient

        return m_gradient,b_gradient

    def plot_graph(self,m : float, b : float) -> None:

        plt.scatter(self.data["YearsExperience"],self.data["Salary"])
        plt.xlabel = "Yrs Of Exp"
        plt.ylabel = "Salary"
        plt.plot(self.data["YearsExperience"],[m * i + b for i in self.data["YearsExperience"]],color = "Red" )
        plt.show()


data : pd.DataFrame = pd.read_csv("Regression/Linear Regression/Salary_dataset.csv")
m_now : float = 0
b_now : float = 0
s : SLR_Gradient_Descent = SLR_Gradient_Descent(data,0.01)

for i in range(300):
    m_now,b_now = s.gradient_desc(m_now,b_now)

s.plot_graph(m_now,b_now)
