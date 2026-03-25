/**
 * Machine Learning Service for Coca-Cola Stock
 * Implements a simple Linear Regression model to predict the next day's close price,
 * and a trading strategy simulation based on those predictions.
 */
class MLService {
    static model = null;
    static simulationResults = null;

    static init() {
        if (!window.appData || !window.appData.history) return;
        this.runPipeline(window.appData.history);
    }

    static runPipeline(historyData) {
        // 1. Prepare Data & Feature Engineering
        const data = [];
        for (let i = 1; i < historyData.length; i++) {
            const prev = historyData[i - 1];
            const curr = historyData[i];
            
            data.push({
                date: curr.ParsedDate,
                dateStr: curr.Date,
                x: prev.Close, // Feature: Previous Day Close
                y: curr.Close, // Target: Current Day Close
                actualRet: (curr.Close / prev.Close) - 1
            });
        }

        // 2. Data Split (80% Train, 20% Test)
        const trainSize = Math.floor(data.length * 0.8);
        const trainData = data.slice(0, trainSize);

        // 3. Modeling - Linear Regression (Ordinary Least Squares)
        this.model = this.trainLinearRegression(trainData);

        // 4. Prediction & Strategy Simulation (on the whole dataset for visualization)
        let stratCumRet = 1.0;
        let marketCumRet = 1.0;

        this.simulationResults = {
            dates: [],
            actualClose: [],
            predictedClose: [],
            strategyReturns: [],
            marketReturns: [],
            stratFinalFactor: 0,
            marketFinalFactor: 0
        };

        for (let i = 0; i < data.length; i++) {
            const row = data[i];
            const y_pred = this.model.m * row.x + this.model.b;
            
            // Trading Strategy: Buy if predicted close > current open (previous close)
            const signal = y_pred > row.x ? 1 : 0;
            const stratRet = signal * row.actualRet;

            stratCumRet *= (1 + stratRet);
            marketCumRet *= (1 + row.actualRet);

            this.simulationResults.dates.push(row.dateStr);
            this.simulationResults.actualClose.push(row.y);
            this.simulationResults.predictedClose.push(y_pred);
            
            // Store cumulative returns as a factor strictly, we'll plot it as 100 * (factor - 1) or just factor
            this.simulationResults.strategyReturns.push(stratCumRet);
            this.simulationResults.marketReturns.push(marketCumRet);
        }

        this.simulationResults.stratFinalFactor = stratCumRet;
        this.simulationResults.marketFinalFactor = marketCumRet;
        
        console.log("ML Pipeline Completed.");
    }

    static trainLinearRegression(data) {
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        const n = data.length;

        for (let i = 0; i < n; i++) {
            const x = data[i].x;
            const y = data[i].y;
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
        }

        const m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const b = (sumY - m * sumX) / n;

        return { m, b };
    }

    /**
     * Get filtered simulation results between two dates
     */
    static getFilteredResults(startDateStr, endDateStr) {
        if (!this.simulationResults) return null;

        const startIdx = this.simulationResults.dates.findIndex(d => d >= startDateStr);
        // Find last element <= endDateStr
        let endIdx = this.simulationResults.dates.length - 1;
        for (let i = this.simulationResults.dates.length - 1; i >= 0; i--) {
            if (this.simulationResults.dates[i] <= endDateStr) {
                endIdx = i;
                break;
            }
        }

        if (startIdx === -1 || startIdx > endIdx) return null;

        // Rebase cumulative returns for the slice window so they start at 1.0 at startIdx
        const baseStrat = this.simulationResults.strategyReturns[startIdx];
        const baseMarket = this.simulationResults.marketReturns[startIdx];

        const sliceStratReturns = this.simulationResults.strategyReturns.slice(startIdx, endIdx + 1).map(v => v / baseStrat);
        const sliceMarketReturns = this.simulationResults.marketReturns.slice(startIdx, endIdx + 1).map(v => v / baseMarket);

        return {
            dates: this.simulationResults.dates.slice(startIdx, endIdx + 1),
            actualClose: this.simulationResults.actualClose.slice(startIdx, endIdx + 1),
            predictedClose: this.simulationResults.predictedClose.slice(startIdx, endIdx + 1),
            strategyReturns: sliceStratReturns,
            marketReturns: sliceMarketReturns,
            stratFinalFactor: sliceStratReturns[sliceStratReturns.length - 1],
            marketFinalFactor: sliceMarketReturns[sliceMarketReturns.length - 1]
        };
    }
}

window.MLService = MLService;
