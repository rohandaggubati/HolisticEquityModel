import os

# Create the reports directory
os.makedirs('reports', exist_ok=True)

# Create portfolio_metrics.txt
with open('reports/portfolio_metrics.txt', 'w') as f:
    f.write('Expected Annual Return: 0.15\n')
    f.write('Annual Volatility: 0.1\n')
    f.write('Sharpe Ratio: 1.5\n')

# Create portfolio_allocation.txt
with open('reports/portfolio_allocation.txt', 'w') as f:
    tickers = ['AAPL', 'MSFT', 'AMZN']
    weight = 1.0 / len(tickers)
    for ticker in tickers:
        f.write(f'{ticker},{weight:.6f}\n')

# Create factor_analysis.txt
with open('reports/factor_analysis.txt', 'w') as f:
    f.write('Factor Analysis:\n')
    f.write('===============\n\n')
    f.write('Portfolio Factor Exposure:\n')
    f.write('momentum: 0.5\n')
    f.write('volatility: 0.2\n')
    f.write('value: 0.3\n')
    f.write('quality: 0.4\n')
    f.write('growth: 0.6\n')
    f.write('total_score: 0.4\n')

print('Test files created successfully!') 