from finvizfinance.screener.overview import Overview

PRESETS = {
    "morning_gappers": {
        # "Country": "USA",
        "Market Cap.": "-Micro (under $300mln)",
        "Price": "Under $20",
        "Average Volume": "Over 500K",
        "Relative Volume": "Over 2",
        "Gap": "Up 5%",
        "Float Short": "Over 10%",
    },
    "secure_Growth": {
        "Market Cap.": "+Large (over $10bln)",
        "Debt/Equity": "Under 1",
        "EPS growthpast 5 years": "Positive (>0%)",
        "Gross Margin": "Over 30%",
        "PEG": "Under 2",
        "Sales growthpast 5 years": "Over 5%",
        "Beta": "Under 1",
        "52-Week High/Low": "20% or more below High",
        "50-Day Simple Moving Average": "Price above SMA50",
    },  
    "seagate_like_setup": {
        "Sales growthqtr over qtr": "Over 20%",
        "Gross Margin": "Over 30%",
        "P/E": "Under 25",
        "PEG": "Under 2",
        # "Performance": "Month Up",
        # "Performance 3M": "Up 10%",
        # "Performance 6M": "Up 25%",
        "50-Day Simple Moving Average": "Price above SMA50",
        "Relative Volume": "Over 1.5",
    },
    "seagate_like_setup_small": {
        "Market Cap.": "+Small (over $300mln)",
        "Average Volume": "Over 500K",
        "Price": "Over $5",
        "P/E": "Under 25",
        "PEG": "Under 2",
        "Sales Growthqtr Over Qtr": "Over 20%",
        "Gross Margin": "Over 30%",
        "50-Day Simple Moving Average": "Price above SMA50",
        "Relative Volume": "Over 1.5",
    }
}   

def use_preset(preset_name):
    return PRESETS.get(preset_name, {})

def main():
    presets = list(PRESETS.keys())
    print("Available presets:")
    for p in presets:
        print(f"  • {p}")
    preset = presets[1]
    foverview = Overview()
    foverview.set_filter(filters_dict=use_preset(preset))
    df = foverview.screener_view(order="Change", limit=100, ascend=False)
    print(df.to_string(index=False))
    df.to_csv("secure_growth.csv", index=False)

if __name__ == "__main__":
    main()