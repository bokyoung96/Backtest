# Backtest Web Interface (bt_web)

A Flask-based web interface for visualizing and analyzing financial statement-based backtesting strategies. This application provides interactive dashboards for performance analysis, transaction cost breakdown, and portfolio holdings visualization.

## Features

- Interactive performance visualization
- Detailed transaction cost analysis
- Portfolio holdings breakdown
- Excel report generation
- Multiple strategy comparison
- Real-time strategy execution

## Project Structure

```
bt_web/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── index.html        # Strategy selection page
│   ├── performance.html  # Strategy performance dashboard
│   ├── about.html        # About page
│   ├── help.html        # Help documentation
│   ├── terms.html       # Terms and conditions
│   ├── privacy.html     # Privacy policy
│   ├── contact.html     # Contact information
│   ├── error.html       # Error page
│   ├── 404.html         # Not found page
│   └── 500.html         # Server error page
├── static/               # Static assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Image assets
└── README.md             # This file
```

## Prerequisites

- Python 3.9 or higher
- bt package (core backtesting package)
- Web browser with JavaScript enabled

## Installation

1. **Install the core package first:**

```bash
cd ../q_root
./setup.sh
```

2. **Install web interface dependencies:**

```bash
cd ../bt_web
pip install -r requirements.txt
```

## Running the Application

1. **Start the Flask server:**

```bash
python app.py
```

2. **Access the web interface:**

- Open your browser and navigate to `http://localhost:5001`
- Default port is 5001 to avoid conflicts with other services

## Available Strategies

### Financial Statement Analysis

- Earnings Change Analysis
- Operating Profit Analysis
- Sales Growth Analysis

Each strategy provides:

- Performance metrics
- Transaction analysis
- Position information
- Risk metrics

## Parameter Configuration

### Basic Settings

- Initial investment amount
- Market selection (e.g., KOSPI200)
- Date range selection
- Rebalancing frequency

### Trading Settings

- Commission rates
- Slippage settings
- Tax rates
- Cash rate

### Strategy-specific Settings

- Quantile settings
- Position weighting
- Factor combinations

## Data Export

Generate detailed Excel reports including:

1. Performance Summary
2. Returns and Drawdown Analysis
3. Transaction Cost Breakdown
4. Cash Balance History
5. Portfolio Holdings

## Development

### Adding New Features

1. Create feature branch
2. Implement changes
3. Test thoroughly
4. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Include docstrings
- Write unit tests

## Security

- Input validation for all parameters
- Error handling for edge cases
- Secure file handling
- Rate limiting for API endpoints

## Browser Compatibility

Tested and supported in:

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
