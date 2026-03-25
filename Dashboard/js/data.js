const DataService = {
    historyData: [],
    infoData: null,
    
    init() {
        return new Promise((resolve) => {
            // Parse Stock History
            const historyCsvData = typeof window.historyStr === 'object' && window.historyStr.value ? window.historyStr.value : window.historyStr;
            Papa.parse(historyCsvData, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    this.historyData = results.data.map(row => {
                        // Ensure date is properly parsed
                        return {
                            ...row,
                            ParsedDate: new Date(row.Date)
                        };
                    }).sort((a, b) => a.ParsedDate - b.ParsedDate); // Sort chronologically
                    
                    // Parse Company Info
                    const infoCsvData = typeof window.infoStr === 'object' && window.infoStr.value ? window.infoStr.value : window.infoStr;
                    Papa.parse(infoCsvData, {
                        header: true,
                        dynamicTyping: true,
                        skipEmptyLines: true,
                        complete: (infoResults) => {
                            this.infoData = infoResults.data[0];
                            resolve();
                        }
                    });
                }
            });
        });
    },

    getFilteredHistory(rangeStr = 'all', startDate = null, endDate = null) {
        if (startDate && endDate) {
            const start = new Date(startDate);
            const end = new Date(endDate);
            return this.historyData.filter(d => d.ParsedDate >= start && d.ParsedDate <= end);
        }

        if (rangeStr === 'all') return this.historyData;
        
        const latestDate = this.historyData[this.historyData.length - 1].ParsedDate;
        let cutoffDate = new Date(latestDate);

        if (rangeStr === '5y') {
            cutoffDate.setFullYear(latestDate.getFullYear() - 5);
        } else if (rangeStr === '1y') {
            cutoffDate.setFullYear(latestDate.getFullYear() - 1);
        } else if (rangeStr === '6m') {
            cutoffDate.setMonth(latestDate.getMonth() - 6);
        }

        return this.historyData.filter(d => d.ParsedDate >= cutoffDate);
    },

    calculateMovingAverage(data, period, key = 'Close') {
        const ma = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                ma.push(null);
            } else {
                let sum = 0;
                for (let j = 0; j < period; j++) {
                    sum += data[i - j][key];
                }
                ma.push(sum / period);
            }
        }
        return ma;
    }
};
