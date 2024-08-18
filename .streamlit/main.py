import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Function for non-dividend paying stocks
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


# Function for dividend paying stocks
def black_scholes_dividend(S, K, T, r, sigma, q, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


# Function to plot the Option Price vs. Time to Maturity (Time Decay Chart)
def plot_time_decay(S, K, T, r, sigma, q=None, option_type="call", dividend=False):
    time_to_maturity = np.linspace(0.01, T, 100)
    option_prices = []

    for t in time_to_maturity:
        if dividend:
            price = black_scholes_dividend(S, K, t, r, sigma, q, option_type)
        else:
            price = black_scholes(S, K, t, r, sigma, option_type)
        option_prices.append(price)

    plt.figure(figsize=(10, 6))
    plt.plot(time_to_maturity, option_prices, label=f'{option_type.capitalize()} Option')
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Option Price')
    plt.title('Option Price vs. Time to Maturity (Time Decay)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def explanation_page():
    st.title("Understanding the Black-Scholes Model")

    st.markdown("""
    The **Black-Scholes model** is a mathematical model used to calculate the theoretical price of European call and put options.
    It assumes that markets are efficient, meaning that price movements are random and cannot be predicted.
    """)

    with st.expander("Click to see the formula"):
        st.latex(r"""
        C = S_0 N(d_1) - K e^{-rT} N(d_2)
        """)
        st.latex(r"""
        d_1 = \frac{\log(S_0 / K) + (r + \sigma^2 / 2) T}{\sigma \sqrt{T}}
        """)
        st.latex(r"""
        d_2 = d_1 - \sigma \sqrt{T}
        """)

    st.markdown("### Explanation of Variables:")
    st.write("""
    - **S_0**: Current stock price
    - **K**: Strike price of the option
    - **T**: Time to maturity in years
    - **r**: Risk-free interest rate
    - **σ (sigma)**: Volatility of the stock's returns
    - **N(x)**: Cumulative distribution function of the standard normal distribution
    """)

    st.markdown("### Derivation Summary")
    st.write("""
    The Black-Scholes equation is derived by solving a partial differential equation that models the price of the option as a function of time and stock price.
    The equation assumes no arbitrage opportunities, constant volatility, and continuous trading of the underlying stock.
    """)

    st.markdown("### Interactivity")
    st.write(
        "To help you understand the impact of different parameters, you can adjust the sliders below and see the results.")

    st.write("**Explore d1 and d2 values:**")
    S = st.slider("Stock Price (S)", 50.0, 150.0, 100.0)
    K = st.slider("Strike Price (K)", 50.0, 150.0, 100.0)
    T = st.slider("Time to Maturity (T in years)", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (r)", 0.01, 0.10, 0.05)
    sigma = st.slider("Volatility (σ)", 0.1, 0.5, 0.2)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    st.write(f"Calculated **d1**: {d1:.4f}")
    st.write(f"Calculated **d2**: {d2:.4f}")

    st.progress(min(max(d1, 0), 1))
    st.success(f"The value of N(d1) is approximately {norm.cdf(d1):.4f}")
    st.info(f"The value of N(d2) is approximately {norm.cdf(d2):.4f}")


def main():
    st.sidebar.title("Option Pricing Calculator")
    page = st.sidebar.selectbox("Choose a page", ["Non-Dividend Paying", "Dividend Paying", "Explanation"])

    if page == "Non-Dividend Paying":
        st.title("Black-Scholes Option Pricing Calculator (Non-Dividend Paying)")
        st.markdown("#### This page focuses on non-dividend paying stocks.")

        S = st.number_input("Stock Price (S)", min_value=0.0, value=100.0, step=0.1)
        K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=0.1)
        T = st.number_input("Time to Maturity (T in years)", min_value=0.0, value=1.0, step=0.01)
        r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
        sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01)
        option_type = st.selectbox("Option Type", ["call", "put"])

        if st.button("Calculate"):
            price = black_scholes(S, K, T, r, sigma, option_type)
            st.write(f"The {option_type} option price is: ${price:.2f}")

        st.markdown("### Option Price vs. Time to Maturity (Time Decay Chart)")
        st.write("The chart below shows how the option price changes as time to maturity decreases.")
        plot_time_decay(S, K, T, r, sigma, option_type=option_type)

    elif page == "Dividend Paying":
        st.title("Black-Scholes Option Pricing Calculator (Dividend Paying)")
        st.markdown("#### This page focuses on dividend paying stocks.")

        S = st.number_input("Stock Price (S)", min_value=0.0, value=100.0, step=0.1)
        K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=0.1)
        T = st.number_input("Time to Maturity (T in years)", min_value=0.0, value=1.0, step=0.01)
        r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
        sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01)
        q = st.number_input("Dividend Yield (q)", min_value=0.0, value=0.03, step=0.01)
        option_type = st.selectbox("Option Type", ["call", "put"])

        if st.button("Calculate"):
            price = black_scholes_dividend(S, K, T, r, sigma, q, option_type)
            st.write(f"The {option_type} option price with dividends is: ${price:.2f}")

        st.markdown("### Option Price vs. Time to Maturity (Time Decay Chart)")
        st.write("The chart below shows how the option price changes as time to maturity decreases.")
        plot_time_decay(S, K, T, r, sigma, q, option_type=option_type, dividend=True)

    elif page == "Explanation":
        explanation_page()


if __name__ == "__main__":
    main()
