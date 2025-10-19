import { useCallback, useEffect, useMemo, useState } from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { Card } from "./ui/card";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Button } from "./ui/button";
import { createCategoryColorLookup, formatCurrency, getCategoryColor } from "../data/financialData";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:8000";

const parseAmount = (value: unknown): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return 0;

  const trimmed = value.trim();
  if (!trimmed) return 0;

  const removeSpaces = trimmed.replace(/[\s\u00A0]/g, "");
  const hasComma = removeSpaces.includes(",");
  const hasDot = removeSpaces.includes(".");

  let normalized = removeSpaces;
  if (hasComma && hasDot) {
    normalized = normalized.replace(/,/g, "");
  } else {
    normalized = normalized.replace(",", ".");
  }

  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : 0;
};

interface BackendTransaction {
  transaction_id: string;
  date: string;
  time: string;
  amount: number;
  amount_money?: number;
  transactioner_id?: string;
  category?: string;
  category_ru?: string;
  item?: string;
  pcs?: number;
  quantity?: number;
}

interface Transaction {
  transactionId: string;
  item: string;
  amount: number;
  category: string;
  categoryRu: string;
  date: string;
  time: string;
  customerId: string;
  quantity: number;
}

interface CategorySummary {
  category: string;
  categoryRu: string;
  totalAmount: number;
  transactionCount: number;
  totalQuantity: number;
  sampleItems: string[];
}

interface TooltipProps {
  active?: boolean;
  payload?: any[];
  totalSpending: number;
}

const CustomTooltip = ({ active, payload, totalSpending }: TooltipProps) => {
  if (!active || !payload || !payload.length) return null;

  const entry = payload[0];
  const percentage =
    totalSpending > 0 ? ((entry.value / totalSpending) * 100).toFixed(1) : "0.0";

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border dark:border-gray-700">
      <p className="font-medium dark:text-white">{entry.name}</p>
      <p className="text-sm text-gray-600 dark:text-gray-400">{formatCurrency(entry.value)}</p>
      <p className="text-xs text-gray-500 dark:text-gray-500">{percentage}% of spending</p>
    </div>
  );
};

export function FinancesTab() {
  const [activeView, setActiveView] = useState<"analytics" | "transactions">("analytics");
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const mapTransaction = (record: BackendTransaction): Transaction => ({
    transactionId: String(record.transaction_id ?? ""),
    item: record.item ?? "Unknown item",
    amount: parseAmount(record.amount_money ?? record.amount ?? 0),
    category: record.category ?? "Other",
    categoryRu: record.category_ru ?? "Прочее",
    date: record.date ?? "",
    time: record.time ?? "",
    customerId: record.transactioner_id ?? "—",
    quantity: Number(record.pcs ?? record.quantity ?? 1) || 1,
  });

  const fetchTransactions = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/get-parsed-transactions`);
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const payload = await response.json();
      const mapped: Transaction[] = (payload.transactions ?? []).map(mapTransaction);
      createCategoryColorLookup(mapped.map((t) => t.category));
      setTransactions(mapped);
    } catch (err) {
      console.error("Failed to load finances data", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchTransactions();
  }, [fetchTransactions]);

  useEffect(() => {
    const handler = () => void fetchTransactions();
    window.addEventListener("zamanbank-transactions-updated", handler);
    return () => window.removeEventListener("zamanbank-transactions-updated", handler);
  }, [fetchTransactions]);

  const totalSpending = useMemo(
    () =>
      transactions.reduce((sum, transaction) => {
        const amount = Number.isFinite(transaction.amount) ? transaction.amount : 0;
        return sum + amount;
      }, 0),
    [transactions]
  );

  const categorySummaries = useMemo(() => {
    const accumulator = new Map<string, CategorySummary>();

    transactions.forEach((transaction) => {
      const entry =
        accumulator.get(transaction.category) ?? {
          category: transaction.category,
          categoryRu: transaction.categoryRu,
          totalAmount: 0,
          transactionCount: 0,
          totalQuantity: 0,
          sampleItems: [] as string[],
        };

      entry.totalAmount += transaction.amount;
      entry.transactionCount += 1;
      entry.totalQuantity += transaction.quantity ?? 0;

      if (transaction.item && !entry.sampleItems.includes(transaction.item)) {
        entry.sampleItems.push(transaction.item);
      }

      accumulator.set(transaction.category, entry);
    });

    return Array.from(accumulator.values()).sort((a, b) => b.totalAmount - a.totalAmount);
  }, [transactions]);

  const topCategories = useMemo(() => categorySummaries.slice(0, 8), [categorySummaries]);

  const chartData = useMemo(
    () =>
      categorySummaries.map((summary) => ({
        name: summary.category,
        value: summary.totalAmount,
        color: getCategoryColor(summary.category),
      })),
    [categorySummaries]
  );

  const handleRefresh = () => void fetchTransactions();

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 px-4 py-4 flex items-center justify-between flex-shrink-0">
        <h1 className="text-lg font-semibold dark:text-white">Finances</h1>
        <Button variant="outline" size="sm" onClick={handleRefresh} className="dark:border-gray-700 dark:text-white">
          Refresh
        </Button>
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
                        Based on the latest {transactions.length} purchases
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold text-[#2D9A86]">{formatCurrency(totalSpending)}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Total spending</p>
                    </div>
                  </div>
                  {error ? (
                    <p className="mt-2 text-sm text-red-600 dark:text-red-400">Failed to load data: {error}</p>
                  ) : null}
                </Card>

                <div className="grid gap-4 lg:grid-cols-2">
                  <Card className="p-4 dark:bg-gray-800 dark:border-gray-700">
                    <h2 className="text-lg font-semibold mb-4 dark:text-white">Category split</h2>
                    <div className="flex justify-center">
                      {isLoading ? (
                        <p className="text-sm text-gray-500 dark:text-gray-400">Loading chart...</p>
                      ) : chartData.length ? (
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
                          <Tooltip content={<CustomTooltip totalSpending={totalSpending} />} />
                        </PieChart>
                      ) : (
                        <p className="text-sm text-gray-500 dark:text-gray-400">No spending data yet.</p>
                      )}
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
                            <span className="inline-flex h-3 w-3 flex-shrink-0 rounded-full" style={{ backgroundColor: color }} />
                            <div className="flex-1">
                              <div className="flex items-center justify-between gap-2 dark:text-white">
                                <span>{summary.category}</span>
                                <span className="text-sm text-gray-500 dark:text-gray-400">{percentage}%</span>
                              </div>
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                {formatCurrency(summary.totalAmount)} · {summary.transactionCount} purchases
                              </p>
                            </div>
                          </div>
                        );
                      })}
                      {!topCategories.length && !isLoading ? (
                        <p className="text-sm text-gray-500 dark:text-gray-400">No category data available.</p>
                      ) : null}
                    </div>
                  </Card>
                </div>

                <div className="space-y-3">
                  <h2 className="text-lg font-semibold dark:text-white">Category details</h2>
                  {categorySummaries.map((summary) => {
                    const percentage =
                      totalSpending > 0 ? ((summary.totalAmount / totalSpending) * 100).toFixed(1) : "0.0";
                    const color = getCategoryColor(summary.category);
                    const itemsLabel = summary.sampleItems.slice(0, 3).join(", ") || "No items captured";

                    return (
                      <Card key={summary.category} className="p-4 dark:bg-gray-800 dark:border-gray-700">
                        <div className="flex items-start gap-3">
                          <div className="mt-1 h-4 w-4 flex-shrink-0 rounded-full" style={{ backgroundColor: color }} />
                          <div className="flex-1">
                            <div className="mb-1 flex items-start justify-between gap-4">
                              <h3 className="font-medium dark:text-white">{summary.category}</h3>
                              <span className="text-sm font-medium dark:text-white">
                                {formatCurrency(summary.totalAmount)}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Top items: {itemsLabel}</p>
                            <div className="w-full rounded-full bg-gray-200 dark:bg-gray-700 h-2">
                              <div
                                className="h-2 rounded-full"
                                style={{ width: `${percentage}%`, backgroundColor: color }}
                              />
                            </div>
                            <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                              {percentage}% of tracked spending · {summary.transactionCount} purchases ·{" "}
                              {summary.totalQuantity} items
                            </div>
                          </div>
                        </div>
                      </Card>
                    );
                  })}
                  {!categorySummaries.length && !isLoading ? (
                    <Card className="p-4 text-sm text-gray-500 dark:text-gray-400 dark:bg-gray-800 dark:border-gray-700">
                      No transactions have been parsed yet. Try adding one from the financial diary tab.
                    </Card>
                  ) : null}
                </div>
              </div>
            </ScrollArea>
          </div>
        </TabsContent>

        <TabsContent value="transactions" className="flex-1 overflow-hidden mt-0">
          <div className="h-full overflow-hidden">
            <ScrollArea className="h-full px-4">
              <div className="py-4 pb-6 space-y-3">
                {isLoading ? (
                  <Card className="p-4 text-sm text-gray-500 dark:text-gray-400 dark:bg-gray-800 dark:border-gray-700">
                    Loading transactions...
                  </Card>
                ) : null}
                {error ? (
                  <Card className="p-4 text-sm text-red-600 dark:text-red-400 dark:bg-gray-800 dark:border-gray-700">
                    Failed to load transactions: {error}
                  </Card>
                ) : null}
                {!transactions.length && !isLoading && !error ? (
                  <Card className="p-4 text-sm text-gray-500 dark:text-gray-400 dark:bg-gray-800 dark:border-gray-700">
                    No transactions recorded yet.
                  </Card>
                ) : null}
                {transactions.map((transaction) => (
                  <Card key={transaction.transactionId} className="p-4 dark:bg-gray-800 dark:border-gray-700">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <div className="font-medium dark:text-white">{transaction.item}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {transaction.date} {transaction.time ? `at ${transaction.time}` : ""}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Customer {transaction.customerId}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-[#2D9A86]">{formatCurrency(transaction.amount)}</div>
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
                      {transaction.categoryRu && (
                        <span className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1 text-xs text-gray-600 dark:border-gray-700 dark:text-gray-300">
                          {transaction.categoryRu}
                        </span>
                      )}
                      {typeof transaction.quantity === "number" && transaction.quantity > 0 ? (
                        <span className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1 text-xs text-gray-600 dark:border-gray-700 dark:text-gray-300">
                          Qty {transaction.quantity}
                        </span>
                      ) : null}
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
