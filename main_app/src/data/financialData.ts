const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

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

const categoryColorLookup = new Map<string, string>();

export const createCategoryColorLookup = (categories: string[]): Map<string, string> => {
  categories.forEach((category) => {
    if (!category) return;
    if (!categoryColorLookup.has(category)) {
      const color = CATEGORY_COLOR_PALETTE[categoryColorLookup.size % CATEGORY_COLOR_PALETTE.length];
      categoryColorLookup.set(category, color);
    }
  });
  return categoryColorLookup;
};

export const getCategoryColor = (category?: string): string => {
  if (!category) return CATEGORY_COLOR_PALETTE[0];
  if (!categoryColorLookup.has(category)) {
    const color = CATEGORY_COLOR_PALETTE[categoryColorLookup.size % CATEGORY_COLOR_PALETTE.length];
    categoryColorLookup.set(category, color);
  }
  return categoryColorLookup.get(category) ?? CATEGORY_COLOR_PALETTE[0];
};

export const formatCurrency = (value: number): string => `${NUMBER_FORMAT.format(value)} KZT`;
