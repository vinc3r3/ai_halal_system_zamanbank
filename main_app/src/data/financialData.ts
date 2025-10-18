import transactionsCsv from '../../zamanbank_transactions.csv?raw';
import financesCsv from '../../zamanbank_finances.csv?raw';

interface RawRecord {
  [key: string]: string;
}

function parseCsv(raw: string): RawRecord[] {
  const lines = raw
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headers = lines[0].split(',').map((header) => header.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(',').map((value) => value.trim());
    const record: RawRecord = {};

    headers.forEach((header, index) => {
      record[header] = values[index] ?? '';
    });

    return record;
  });
}

const rawTransactions = parseCsv(transactionsCsv);
const rawFinances = parseCsv(financesCsv);

const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const toNumber = (value: string): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const toQuantity = (value: string): number | undefined => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const toTimestamp = (date: string, time: string): number => {
  const [month, day, year] = date.split('/').map((part) => Number(part));
  const [hours, minutes, seconds] = time.split(':').map((part) => Number(part));

  if (
    !Number.isFinite(year) ||
    !Number.isFinite(month) ||
    !Number.isFinite(day) ||
    !Number.isFinite(hours) ||
    !Number.isFinite(minutes)
  ) {
    return 0;
  }

  const safeSeconds = Number.isFinite(seconds) ? seconds : 0;
  return Date.UTC(year, month - 1, day, hours, minutes, safeSeconds);
};

export interface TransactionRecord {
  transactionId: string;
  date: string;
  time: string;
  amount: number;
  customerId: string;
  occurredAt: number;
}

export interface FinanceRecord {
  transactionId: string;
  category: string;
  amount: number;
  item: string;
  quantity?: number;
}

export interface EnrichedTransaction extends TransactionRecord {
  category?: string;
  item?: string;
  quantity?: number;
  hasReceipt: boolean;
}

export interface CategorySummary {
  category: string;
  totalAmount: number;
  transactionCount: number;
  totalQuantity: number;
  sampleItems: string[];
}

export const CATEGORY_COLOR_PALETTE = [
  '#2D9A86',
  '#EEFE6D',
  '#FF6B6B',
  '#4ECDC4',
  '#95E1D3',
  '#F38181',
  '#AA96DA',
  '#FCBAD3',
  '#A8E6CF',
  '#FFD93D',
  '#6BCB77',
  '#4D96FF',
  '#FF9A76',
  '#B388EB',
  '#FEC7D7',
  '#FFA5AB',
] as const;

export const transactions: TransactionRecord[] = rawTransactions.map((record) => {
  const transactionId = record.transaction_id ?? '';
  const date = record.date ?? '';
  const time = record.time ?? '';
  const amount = toNumber(record.amount ?? '');

  return {
    transactionId,
    date,
    time,
    amount,
    customerId: record.transactioner_id ?? '',
    occurredAt: toTimestamp(date, time),
  };
});

export const finances: FinanceRecord[] = rawFinances.map((record) => ({
  transactionId: record.transaction_id ?? '',
  category: record.category ?? 'Uncategorized',
  amount: toNumber(record.amount_money ?? ''),
  item: record.item ?? 'Purchase',
  quantity: toQuantity(record.pcs ?? ''),
}));

const financeByTransaction = new Map(finances.map((finance) => [finance.transactionId, finance]));

const baseEnrichedTransactions: EnrichedTransaction[] = transactions.map((transaction) => {
  const finance = financeByTransaction.get(transaction.transactionId);

  return {
    ...transaction,
    amount: finance?.amount ?? transaction.amount,
    category: finance?.category ?? 'Uncategorized',
    item: finance?.item,
    quantity: finance?.quantity,
    hasReceipt: Boolean(finance),
  };
});

const demoTransactions: EnrichedTransaction[] = [
  {
    transactionId: 'demo-001',
    date: '6/20/2025',
    time: '08:45:00',
    amount: 4200,
    customerId: 'CUST-DEMO01',
    occurredAt: toTimestamp('6/20/2025', '08:45:00'),
    category: 'Uncategorized',
    item: 'Morning Coffee Run',
    quantity: 2,
    hasReceipt: false,
  },
  {
    transactionId: 'demo-002',
    date: '6/18/2025',
    time: '19:20:00',
    amount: 12800,
    customerId: 'CUST-DEMO02',
    occurredAt: toTimestamp('6/18/2025', '19:20:00'),
    category: 'Transport',
    item: 'Ride Share',
    quantity: 1,
    hasReceipt: false,
  },
  {
    transactionId: 'demo-003',
    date: '6/16/2025',
    time: '13:05:00',
    amount: 9800,
    customerId: 'CUST-DEMO03',
    occurredAt: toTimestamp('6/16/2025', '13:05:00'),
    category: 'Uncategorized',
    item: 'Team Lunch',
    quantity: 4,
    hasReceipt: false,
  },
];

export const enrichedTransactions: EnrichedTransaction[] = [...baseEnrichedTransactions, ...demoTransactions].sort(
  (a, b) => b.occurredAt - a.occurredAt,
);

const categorySummaryMap = new Map<string, CategorySummary>();

for (const finance of finances) {
  const entry =
    categorySummaryMap.get(finance.category) ??
    {
      category: finance.category,
      totalAmount: 0,
      transactionCount: 0,
      totalQuantity: 0,
      sampleItems: [] as string[],
    };

  entry.totalAmount += finance.amount;
  entry.transactionCount += 1;
  entry.totalQuantity += finance.quantity ?? 0;

  if (finance.item && !entry.sampleItems.includes(finance.item)) {
    entry.sampleItems.push(finance.item);
  }

  categorySummaryMap.set(finance.category, entry);
}

export const categorySummaries: CategorySummary[] = Array.from(categorySummaryMap.values()).sort(
  (a, b) => b.totalAmount - a.totalAmount,
);

export const totalSpending = categorySummaries.reduce((sum, entry) => sum + entry.totalAmount, 0);

export const formatCurrency = (value: number): string => `${NUMBER_FORMAT.format(value)} KZT`;

export const categoryColorLookup = new Map(
  categorySummaries.map((summary, index) => [
    summary.category,
    CATEGORY_COLOR_PALETTE[index % CATEGORY_COLOR_PALETTE.length],
  ]),
);

export const getCategoryColor = (category?: string): string => {
  if (!category) {
    return CATEGORY_COLOR_PALETTE[0];
  }
  return categoryColorLookup.get(category) ?? CATEGORY_COLOR_PALETTE[0];
};
