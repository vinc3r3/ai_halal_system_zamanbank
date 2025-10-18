import { useState } from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { Card } from "./ui/card";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  categorySummaries,
  enrichedTransactions,
  formatCurrency,
  totalSpending,
  getCategoryColor,
} from "../data/financialData";

const chartData = categorySummaries.map((summary) => ({
  name: summary.category,
  value: summary.totalAmount,
  color: getCategoryColor(summary.category),
}));

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const entry = payload[0];
    const percentage =
      totalSpending > 0 ? ((entry.value / totalSpending) * 100).toFixed(1) : "0.0";

    return (
      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border dark:border-gray-700">
        <p className="font-medium dark:text-white">{entry.name}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {formatCurrency(entry.value)}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-500">{percentage}% of spending</p>
      </div>
    );
  }
  return null;
};

export function FinancesTab() {
  const [activeView, setActiveView] = useState<"analytics" | "transactions">("analytics");

  const sortedSummaries = categorySummaries.slice().sort((a, b) => b.totalAmount - a.totalAmount);
  const topCategories = sortedSummaries.slice(0, 8);

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 px-4 py-4 flex-shrink-0">
        <h1 className="text-lg font-semibold dark:text-white">Finances</h1>
      </div>

      <Tabs
        value={activeView}
        onValueChange={(value) => setActiveView(value as "analytics" | "transactions")}
        className="flex-1 flex flex-col overflow-hidden"
      >
        <TabsList className="mx-4 mt-4 grid w-auto grid-cols-2 dark:bg-gray-800 flex-shrink-0">
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
        </TabsList>

        <TabsContent value="analytics" className="flex-1 overflow-hidden mt-0">
          <div className="h-full overflow-hidden">
            <ScrollArea className="h-full px-4">
              <div className="py-4 pb-6 space-y-4">
              <Card className="p-4 dark:bg-gray-800 dark:border-gray-700">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold dark:text-white">Spending overview</h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Based on the latest {enrichedTransactions.length} purchases
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-[#2D9A86]">
                      {formatCurrency(totalSpending)}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Total spending</p>
                  </div>
                </div>
              </Card>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="p-4 dark:bg-gray-800 dark:border-gray-700">
                  <h2 className="text-lg font-semibold mb-4 dark:text-white">Category split</h2>
                  <div className="flex justify-center">
                    <PieChart width={320} height={256}>
                      <Pie
                        data={chartData}
                        cx="50%"
                        cy="50%"
                        dataKey="value"
                        innerRadius={60}
                        outerRadius={90}
                        paddingAngle={4}
                      >
                        {chartData.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </div>
                </Card>

                <Card className="p-4 dark:bg-gray-800 dark:border-gray-700">
                  <h2 className="text-lg font-semibold mb-4 dark:text-white">Top categories</h2>
                  <div className="space-y-3">
                    {topCategories.map((summary) => {
                      const percentage =
                        totalSpending > 0
                          ? ((summary.totalAmount / totalSpending) * 100).toFixed(1)
                          : "0.0";
                      const color = getCategoryColor(summary.category);

                      return (
                        <div key={summary.category} className="flex items-center gap-2">
                          <div
                            className="h-3 w-3 rounded-full flex-shrink-0"
                            style={{ backgroundColor: color }}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm truncate dark:text-white">{summary.category}</div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {percentage}% of spending
                            </div>
                          </div>
                          <div className="text-sm font-medium dark:text-white">
                            {formatCurrency(summary.totalAmount)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              </div>

              <div className="space-y-3">
                <h2 className="text-lg font-semibold dark:text-white">Category details</h2>
                {sortedSummaries.map((summary) => {
                  const percentage =
                    totalSpending > 0
                      ? ((summary.totalAmount / totalSpending) * 100).toFixed(1)
                      : "0.0";
                  const color = getCategoryColor(summary.category);
                  const itemsLabel =
                    summary.sampleItems.slice(0, 3).join(", ") || "No items captured";

                  return (
                    <Card key={summary.category} className="p-4 dark:bg-gray-800 dark:border-gray-700">
                      <div className="flex items-start gap-3">
                        <div
                          className="mt-1 h-4 w-4 flex-shrink-0 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                        <div className="flex-1">
                          <div className="mb-1 flex items-start justify-between gap-4">
                            <h3 className="font-medium dark:text-white">{summary.category}</h3>
                            <span className="text-sm font-medium dark:text-white">
                              {formatCurrency(summary.totalAmount)}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                            Top items: {itemsLabel}
                          </p>
                          <div className="w-full rounded-full bg-gray-200 dark:bg-gray-700 h-2">
                            <div
                              className="h-2 rounded-full"
                              style={{ width: `${percentage}%`, backgroundColor: color }}
                            />
                          </div>
                          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                            {percentage}% of tracked spending • {summary.transactionCount} purchases • {summary.totalQuantity} items
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
              </div>
            </ScrollArea>
          </div>
        </TabsContent>

        <TabsContent value="transactions" className="flex-1 overflow-hidden mt-0">
          <div className="h-full overflow-hidden">
            <ScrollArea className="h-full px-4">
              <div className="py-4 pb-6 space-y-3">
              {enrichedTransactions.map((transaction) => (
                <Card
                  key={transaction.transactionId}
                  className="p-4 dark:bg-gray-800 dark:border-gray-700"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="font-medium dark:text-white">
                        {transaction.item ?? "Purchase"}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {transaction.date} at {transaction.time}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Customer {transaction.customerId}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-[#2D9A86]">
                        -{formatCurrency(transaction.amount)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        #{transaction.transactionId.slice(0, 8).toUpperCase()}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {transaction.category && (
                      <span className="inline-flex items-center rounded-full bg-[#EEFE6D] px-3 py-1 text-xs font-medium text-gray-900">
                        {transaction.category}
                      </span>
                    )}
                    {typeof transaction.quantity === "number" && (
                      <span className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1 text-xs text-gray-600 dark:border-gray-700 dark:text-gray-300">
                        Qty {transaction.quantity}
                      </span>
                    )}
                  </div>
                </Card>
              ))}
              </div>
            </ScrollArea>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
