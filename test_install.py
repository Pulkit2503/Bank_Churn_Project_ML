try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    import joblib
    print("✅ All packages installed successfully!")
    print("🎉 You're ready to build the project!")
except ImportError as e:
    print(f"❌ Error: {e}")