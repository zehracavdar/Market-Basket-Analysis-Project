# Market-Basket-Analysis-Project

Project files are presented in the `market_basket_project/` folder. The functions of these files are:

*   **`main.py` (Main Code):** The brain of the project. Loads data, processes it, draws charts, runs the model, and saves results.
*   **`src/preprocessing.py` (Data Preparation):** Contains functions that read the raw CSV file, turn customer baskets into lists, and convert them to a 0-1 matrix.
*   **`src/eda.py` (Analysis/Chart):** Code that calculates best-selling products and draws the `top_products_output.png` chart.
*   **`src/model.py` (AI Model):** Code that runs the FP-Growth algorithm and calculates rules (Lift, Confidence).
*   **`market_basket_results.csv` (Result File):** List of all found rules openable in Excel.
*   **`top_products_output.png` (Image):** Chart of the 15 most popular products.
