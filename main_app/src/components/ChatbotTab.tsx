import { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Mic,
  Send,
  Paperclip,
  X,
  Menu,
  Sparkles,
  FileText,
  Search,
  File,
  Wallet,
  ChevronLeft,
  Loader2,
  Volume2,
  Tag,
  MessageCircle,
} from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { useTheme } from '../contexts/ThemeContext';
import { createCategoryColorLookup, formatCurrency, getCategoryColor } from '../data/financialData';

// ---------- Types ----------
interface ChatOption {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  badge?: string;
}

interface CitationInfo {
  id?: string;
  chapter?: string;
  topic?: string;
  explanation?: string;
  source?: string;
  type?: string;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  citations?: CitationInfo[];
  attachment?: { type: 'image' | 'audio' | 'file'; url: string };
}

interface DiaryTransaction {
  transactionId: string;
  item: string | null;
  amount: number;
  category: string | null;
  categoryRu: string | null;
  date: string;
  time: string;
  customerId: string | null;
  quantity: number | null;
}

// ---------- Style options ----------
const chatOptions: ChatOption[] = [
  { id: 'financial-diary', title: 'Финансовый дневник', description: 'Отслеживайте расходы и доходы', icon: <Wallet className="w-6 h-6" /> },
  { id: 'zaman-ai', title: 'Zaman AI', description: 'Как взять кредит, копить и тратить, не нарушая законов Шариата', icon: <Sparkles className="w-6 h-6" />, badge: 'Халяль' },
  { id: 'text-work', title: 'Работа с текстом', description: 'Пишет за вас, подсказывает идеи', icon: <FileText className="w-6 h-6" />, badge: 'GPT-4o бесплатно' },
  { id: 'ai-search', title: 'ИИ-поисковик', description: 'Ищет ответы на любые вопросы', icon: <Search className="w-6 h-6" /> },
  { id: 'file-work', title: 'Работа с файлами', description: 'Ищет важное в файлах и объясняет простыми словами', icon: <File className="w-6 h-6" /> },
];

// ---------- ENV / helpers ----------
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:8000';

const TIMESTAMP_LOCALE: Intl.LocalesArgument = 'ru-RU';
const formatTimestamp = () =>
  new Date().toLocaleTimeString(TIMESTAMP_LOCALE, { hour: '2-digit', minute: '2-digit' });

const detectLanguage = (text: string): 'ru' | 'en' => (/[\u0400-\u04FF]/.test(text) ? 'ru' : 'en');

const isHexColorLight = (hex: string): boolean => {
  const sanitized = hex.replace('#', '');
  if (sanitized.length !== 6) return false;
  const r = parseInt(sanitized.slice(0, 2), 16);
  const g = parseInt(sanitized.slice(2, 4), 16);
  const b = parseInt(sanitized.slice(4, 6), 16);
  if ([r, g, b].some((value) => Number.isNaN(value))) return false;
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.6;
};

const toNumber = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const sanitized = value
      .trim()
      .replace(/\s+/g, '')
      .replace(/\u00a0/g, '')
      .replace(/,/g, '.')
      .replace(/[^\d.-]/g, '');
    const parsed = Number(sanitized);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
};

const toQuantity = (value: unknown): number | null => {
  const parsed = Math.round(toNumber(value));
  return parsed > 0 ? parsed : null;
};

const normalizeText = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length ? trimmed : null;
};

const parseTimestamp = (date: string | null, time: string | null): number => {
  const normalizedDate = date ? date.trim() : '';
  const normalizedTime = time ? time.trim() : '';
  if (!normalizedDate && !normalizedTime) return 0;
  const composed = `${normalizedDate} ${normalizedTime}`.trim();
  const timestamp = Date.parse(composed);
  return Number.isNaN(timestamp) ? 0 : timestamp;
};

// ---------- Fixed strings ----------
const GENERAL_ERROR = 'Не удалось связаться с Zaman AI. Повторите попытку позже.';
const TRANSCRIPTION_ERROR = 'Не удалось распознать голосовое сообщение. Попробуйте ещё раз.';
const MICROPHONE_ERROR = 'Нет доступа к микрофону. Проверьте разрешения и повторите попытку.';
const AUDIO_ERROR = 'Не удалось воспроизвести аудиоответ. Попробуйте ещё раз.';
const VOICE_UNAVAILABLE_ERROR = 'Голосовые функции недоступны для текущей конфигурации.';

// ---------- Utilities ----------
const extractChatReply = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') return null;
  const data = payload as Record<string, unknown>;
  const candidates = [data.response, data.reply, data.message];
  for (const c of candidates) if (typeof c === 'string' && c.trim()) return c.trim();
  return null;
};

const parseCitationsFromPayload = (payload: unknown): CitationInfo[] => {
  if (!payload || typeof payload !== 'object') return [];
  const citations = (payload as Record<string, unknown>).citations;
  if (!Array.isArray(citations)) return [];

  return citations
    .map((entry) => {
      if (!entry || typeof entry !== 'object') return null;
      const record = entry as Record<string, unknown>;
      const chapter = typeof record.chapter === 'string' ? record.chapter.trim() : undefined;
      const explanation = typeof record.explanation === 'string' ? record.explanation.trim() : undefined;
      const topic = typeof record.topic === 'string' ? record.topic.trim() : undefined;
      const id = typeof record.id === 'string' ? record.id.trim() : undefined;
      const source = typeof record.source === 'string' ? record.source : undefined;
      const type = typeof record.type === 'string' ? record.type : undefined;

      if (!chapter && !id && !explanation && !topic) return null;

      return {
        id,
        chapter,
        topic,
        explanation,
        source,
        type,
      } as CitationInfo;
    })
    .filter((entry): entry is CitationInfo => Boolean(entry));
};

const normalizeCitationKey = (value: string): string =>
  value
    .replace(/\u00a0/g, ' ')
    .replace(/[–—]/g, '-')
    .replace(/^[\s"'«»„”“()]+|[\s"'«»„”“)]+$/g, '')
    .replace(/\s+/g, ' ')
    .replace(/[.,;:]+$/g, '')
    .trim()
    .toLowerCase();

const CITATION_PATTERN = /\(\s*{cite:\s*([^}]+)}\s*\)|\(([^()]+)\)/g;

const decodeBase64Audio = (base64Audio: string): Uint8Array => {
  const binary = atob(base64Audio);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
};

const fetchAudioResponse = async (text: string, language: string): Promise<Blob> => {
  const endpoints = [`${API_BASE_URL}/tts`, `${API_BASE_URL}/speech`];
  const payload = JSON.stringify({ text, language });
  let lastError: Error | null = null;

  for (const endpoint of endpoints) {
    try {
      const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload });
      if (!res.ok) {
        if (res.status === 404) continue;
        let detail: string | undefined;
        try {
          const errBody = await res.json();
          if (typeof errBody?.detail === 'string') detail = errBody.detail;
        } catch {}
        throw new Error(detail ?? 'Zaman AI сейчас недоступен. Попробуйте позже.');
      }
      const ct = res.headers.get('content-type') ?? '';
      if (ct.includes('application/json')) {
        const data: any = await res.json();
        const audioBase64 = data.audio_base64 || data.audioBase64 || data.audio;
        if (!audioBase64) {
          const detail = data.detail || data.message;
          throw new Error(detail ?? 'Получен неожиданный аудиоответ от Zaman AI.');
        }
        const mimeType = data.mime_type || data.mimeType || 'audio/mpeg';
        return new Blob([decodeBase64Audio(audioBase64)], { type: mimeType });
      }
      return await res.blob();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(GENERAL_ERROR);
    }
  }
  throw lastError ?? new Error(GENERAL_ERROR);
};

// ---------- Component ----------
export function ChatbotTab() {
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isUploadingReceipt, setIsUploadingReceipt] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isParsingText, setIsParsingText] = useState(false);
  const [transactions, setTransactions] = useState<DiaryTransaction[]>([]);
  const scrollViewportRef = useRef<HTMLDivElement>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioSourceRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);

  const { theme } = useTheme(); // preserved

  // Function to scroll to bottom
  const scrollToBottom = () => {
    if (scrollViewportRef.current) {
      scrollViewportRef.current.scrollTop = scrollViewportRef.current.scrollHeight;
    }
  };

  const buildCitationMap = (citations?: CitationInfo[]) => {
    const map = new Map<string, { citation: CitationInfo; label: string }>();
    if (!citations) return map;

    citations.forEach((citation) => {
      if (!citation) return;
      const labels = [citation.chapter, citation.id].filter((value): value is string => Boolean(value && value.trim()));
      labels.forEach((label) => {
        const normalized = normalizeCitationKey(label);
        if (!normalized || map.has(normalized)) return;
        const displayLabel = (citation.chapter ?? label).trim() || label.trim();
        map.set(normalized, { citation, label: displayLabel });
      });
    });

    return map;
  };

  const renderCitationBadge = (citation: CitationInfo, displayLabel: string, idx: number) => (
    <Tooltip key={`citation-${displayLabel}-${idx}`}>
      <TooltipTrigger asChild>
        <span className="ml-1 inline-flex items-center gap-1 rounded-full border border-[#2D9A86]/40 bg-[#2D9A86]/10 px-2 py-0.5 text-xs font-medium text-[#2D9A86]">
          <Tag className="h-3 w-3" />
          {displayLabel}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs whitespace-pre-wrap text-xs leading-relaxed">
        {citation.topic ? <p className="mb-1 font-semibold">{citation.topic}</p> : null}
        <p>{citation.explanation ?? 'Комментарий отсутствует в источнике.'}</p>
      </TooltipContent>
    </Tooltip>
  );

  const transformMessageWithCitations = (text: string, citations?: CitationInfo[]): (string | JSX.Element)[] => {
    if (!text) return [];
    const citationMap = buildCitationMap(citations);
    if (!citationMap.size) return [text];

    const nodes: (string | JSX.Element)[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    let citationIndex = 0;
    CITATION_PATTERN.lastIndex = 0;

    while ((match = CITATION_PATTERN.exec(text)) !== null) {
      if (match.index > lastIndex) {
        nodes.push(text.slice(lastIndex, match.index));
      }

      const rawLabel = (match[1] ?? match[2] ?? '').trim();
      const normalized = normalizeCitationKey(rawLabel);
      const entry = normalized ? citationMap.get(normalized) : undefined;

      if (entry) {
        nodes.push(renderCitationBadge(entry.citation, entry.label || rawLabel, citationIndex));
        citationIndex += 1;
      } else {
        nodes.push(match[0]);
      }

      lastIndex = CITATION_PATTERN.lastIndex;
    }

    if (lastIndex < text.length) {
      nodes.push(text.slice(lastIndex));
    }

    return nodes;
  };

  const renderMessageContent = (msg: Message) => {
    const baseClass = 'whitespace-pre-wrap leading-relaxed';
    if (msg.sender !== 'ai') {
      return <p className={baseClass}>{msg.text}</p>;
    }

    const segments = transformMessageWithCitations(msg.text, msg.citations);
    if (!segments.length) {
      return <p className={`${baseClass} text-sm`}>{msg.text}</p>;
    }

    return (
      <div className={`${baseClass} text-sm`}>
        {segments.map((segment, idx) =>
          typeof segment === 'string' ? <span key={`ai-text-${msg.id}-${idx}`}>{segment}</span> : segment
        )}
      </div>
    );
  };

  // cleanup
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch {}
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
      }
      if (audioSourceRef.current) {
        URL.revokeObjectURL(audioSourceRef.current);
        audioSourceRef.current = null;
      }
    };
  }, []);

  // reset errors when switching chat
  useEffect(() => { setError(null); }, [selectedChat]);

  // Load parsed transactions from CSV when financial diary is opened
  useEffect(() => {
    if (selectedChat === 'financial-diary') {
      void loadTransactions();
    }
  }, [selectedChat]);

  // Auto-scroll to bottom when financial diary is opened or new transactions are added
  useEffect(() => {
    if (selectedChat === 'financial-diary') {
      setTimeout(scrollToBottom, 100);
    }
  }, [selectedChat, transactions]);

  // Load parsed transactions from backend
  const mapRawTransaction = (raw: any): DiaryTransaction | null => {
    const transactionId = normalizeText(raw?.transaction_id) ?? '';
    if (!transactionId) return null;

    const item = normalizeText(raw?.item);
    const category = normalizeText(raw?.category);
    const categoryRu = normalizeText(raw?.category_ru);
    const date = normalizeText(raw?.date) ?? '';
    const time = normalizeText(raw?.time) ?? '';
    const customerId = normalizeText(raw?.transactioner_id);
    const amount = toNumber(raw?.amount_money ?? raw?.amount);
    const quantity = toQuantity(raw?.pcs ?? raw?.quantity);

    return {
      transactionId,
      item,
      amount,
      category,
      categoryRu,
      date,
      time,
      customerId,
      quantity,
    };
  };

  const loadTransactions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-parsed-transactions`);
      if (!response.ok) {
        console.error('⚠️ Failed to load transactions:', response.status);
        return;
      }

      const payload = await response.json();
      const mapped: DiaryTransaction[] = (payload?.transactions ?? [])
        .map(mapRawTransaction)
        .filter(Boolean) as DiaryTransaction[];

      mapped.sort((a, b) => parseTimestamp(a.date, a.time) - parseTimestamp(b.date, b.time));
      createCategoryColorLookup(mapped.map((tx) => tx.category ?? '').filter(Boolean));

      if (isMountedRef.current) {
        setTransactions(mapped);
      }
    } catch (error) {
      console.error('❌ Error loading transactions:', error);
    }
  };

  // -------- chat submit --------
  const mapHistory = (history: Message[]) =>
    history.map((m) => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text }));

  const submitMessage = async (content: string) => {
    const trimmed = content.trim();
    if (!trimmed || isProcessing) return;

    const userMessage: Message = { id: Date.now().toString(), text: trimmed, sender: 'user', timestamp: formatTimestamp() };
    setMessages((prev) => [...prev, userMessage]);
    setError(null);
    setIsProcessing(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, history: mapHistory(messages) }),
      });

      if (!res.ok) {
        if (res.status === 404) throw new Error(VOICE_UNAVAILABLE_ERROR);
        let detail = 'Zaman AI сейчас недоступен. Попробуйте позже.';
        try {
          const body = await res.json();
          if (typeof (body as any)?.detail === 'string') detail = (body as any).detail;
        } catch {}
        throw new Error(detail);
      }

      const data = await res.json();
      const aiText = extractChatReply(data);
      if (!aiText) throw new Error('Неожиданный ответ от Zaman AI.');
      const citations = parseCitationsFromPayload(data);
      const aiMessage: Message = {
        id: `${Date.now()}-ai`,
        text: aiText,
        sender: 'ai',
        timestamp: formatTimestamp(),
        citations: citations.length ? citations : undefined,
      };
      if (isMountedRef.current) setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      // Fallback message if fetch failed (e.g., CORS/offline)
      if (err instanceof Error && (err.message.includes('fetch') || err.message.includes('Failed to fetch'))) {
        const fallbackMessage: Message = {
          id: `${Date.now()}-ai-fallback`,
          text: 'Извините, Zaman AI временно недоступен. Пожалуйста, попробуйте позже или обратитесь в службу поддержки.',
          sender: 'ai',
          timestamp: formatTimestamp(),
        };
        if (isMountedRef.current) setMessages((prev) => [...prev, fallbackMessage]);
      } else {
        if (isMountedRef.current) setError(err instanceof Error ? err.message : GENERAL_ERROR);
      }
    } finally {
      if (isMountedRef.current) setIsProcessing(false);
    }
  };

  const handleSend = () => {
    const trimmed = message.trim();
    if (!trimmed) return;
    setMessage('');
    if (selectedChat === 'financial-diary') {
      handleFinancialDiaryMessage(trimmed);
    } else {
      void submitMessage(trimmed);
    }
  };

  // Heuristic parser (kept if you want local parsing later; not used by API flow)
  const parseFinancialText = (text: string) => {
    const lowerText = text.toLowerCase();
    const amountPatterns = [
      /(\d+(?:[.,]\d{1,2})?)\s*₸/,
      /(\d+(?:[.,]\d{1,2})?)\s*тенге/,
      /(\d+(?:[.,]\d{1,2})?)\s*тг/,
      /(\d+(?:[.,]\d{1,2})?)\s*руб/,
      /(\d+(?:[.,]\d{1,2})?)\s*₽/,
      /(\d+(?:[.,]\d{1,2})?)\s*долларов?/,
      /(\d+(?:[.,]\d{1,2})?)\s*евро/,
      /(\d+(?:[.,]\d{1,2})?)/,
    ];
    let amount = 0;
    for (const p of amountPatterns) {
      const m = lowerText.match(p);
      if (m) { amount = Number(m[1].replace(',', '.')); break; }
    }

    const itemPatterns = [
      /(?:купил|купила|потратил|потратила|заплатил|заплатила)\s+(.+?)\s+(?:за|на|в)/,
      /(?:покупка|расход|трата)\s+(.+?)\s+(?:за|на|в)/,
      /(?:я|мы)\s+(?:купил|купила|потратил|потратила|заплатил|заплатила)\s+(.+?)\s+(?:за|на|в)/,
    ];
    let item = '';
    for (const p of itemPatterns) {
      const m = lowerText.match(p);
      if (m) { item = m[1].trim(); break; }
    }
    if (!item && amount > 0) {
      const words = text.split(/\s+/);
      const idx = words.findIndex(w =>
        w.includes(String(amount)) || /₸|тенге|тг|руб|₽/i.test(w)
      );
      if (idx > 0) {
        item = words.slice(0, idx).join(' ')
          .replace(/\b(купил|купила|потратил|потратила|заплатил|заплатила|я|мы)\b/gi, '')
          .trim();
      }
    }
    if (!item && amount > 0) {
      const clean = text
        .replace(/\d+(?:[.,]\d{1,2})?\s*[₸₽]/gi, '')
        .replace(/\d+(?:[.,]\d{1,2})?\s*(тенге|тг|руб|долларов?|евро)/gi, '')
        .replace(/\b(купил|купила|потратил|потратила|заплатил|заплатила|я|мы|за|на|в|сегодня|вчера)\b/gi, '')
        .trim();
      if (clean) item = clean;
    }

    const categories: Record<string, string[]> = {
      'Продукты': ['еда', 'продукты', 'еду', 'кока', 'кола', 'пепси', 'хлеб', 'молоко', 'мясо', 'рыба', 'овощи', 'фрукты', 'магазин', 'супермаркет', 'продуктовый', 'кофе', 'чай', 'сок', 'вода', 'напиток', 'напитки'],
      'Транспорт': ['такси', 'автобус', 'метро', 'транспорт', 'бензин', 'топливо', 'парковка', 'проезд', 'машина', 'автомобиль'],
      'Утилиты': ['электричество', 'свет', 'газ', 'вода', 'отопление', 'интернет', 'телефон', 'связь', 'коммунальные', 'услуги'],
      'Развлечения': ['кино', 'театр', 'кафе', 'ресторан', 'клуб', 'игра', 'игры', 'развлечения', 'концерт', 'музей'],
      'Здоровье': ['лекарства', 'аптека', 'врач', 'больница', 'медицина', 'здоровье', 'лечение', 'анализы'],
      'Одежда': ['одежда', 'обувь', 'магазин одежды', 'шопинг', 'платье', 'рубашка', 'джинсы'],
      'Образование': ['книги', 'курсы', 'обучение', 'школа', 'университет', 'учебники', 'образование'],
      'Другое': [],
    };
    const itemLower = item.toLowerCase();
    let category = 'Другое';
    for (const [cat, kws] of Object.entries(categories)) {
      if (kws.some(k => itemLower.includes(k))) { category = cat; break; }
    }

    return { amount, item: item || 'Покупка', category, success: amount > 0 };
  };

  const handleFinancialDiaryMessage = async (content: string) => {
    console.log('🔍 Processing financial diary message:', content);
    setIsParsingText(true);

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      text: content,
      sender: 'user',
      timestamp: formatTimestamp(),
    };
    setMessages((prev) => [...prev, userMessage]);

    const processingMessage: Message = {
      id: `processing-${Date.now()}`,
      text: 'Обрабатываю ваш запрос...',
      sender: 'ai',
      timestamp: formatTimestamp(),
    };
    setMessages((prev) => [...prev, processingMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/parse-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: content }),
      });

      setMessages((prev) => prev.filter((m) => m.id !== processingMessage.id));

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API request failed: ${response.status} - ${errorText}`);
      }

      const parsed = await response.json();

      if (parsed.success) {
        const newTransaction: DiaryTransaction & { balance: number | null } = {
          transactionId: `parsed-${Date.now()}`,
          item: parsed.item,
          amount: parsed.amount,
          category: parsed.category,      // keep original key if backend sends it
          categoryRu: parsed.category_ru, // RU label
          date: new Date().toLocaleDateString('ru-RU'),
          time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          customerId: null,
          quantity: 1,
          balance: null,
        };

        try {
          const saveResponse = await fetch(`${API_BASE_URL}/save-transaction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newTransaction),
          });

          if (saveResponse.ok) {
            await saveResponse.json();
            setTransactions((prev) => [...prev, newTransaction]); // append so newest sits at the bottom
            setTimeout(scrollToBottom, 200);
          } else {
            throw new Error('Failed to save transaction');
          }
        } catch (saveErr) {
          console.error('Error saving transaction:', saveErr);
          throw saveErr;
        }

        const successMessage: Message = {
          id: `success-${Date.now()}`,
          text: `✓ Добавлено: ${parsed.item} за ${Number(parsed.amount).toLocaleString('ru-RU', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })} ₸`,
          sender: 'ai',
          timestamp: formatTimestamp(),
        };
        setMessages((prev) => [...prev, successMessage]);
      } else {
        const errorMessage: Message = {
          id: `error-${Date.now()}`,
          text:
            parsed.error_message ||
            'Не удалось распознать сумму или товар в вашем сообщении. Попробуйте написать в формате: "купил хлеб за 200 тенге" или "потратил 500 ₸ на кофе".',
          sender: 'ai',
          timestamp: formatTimestamp(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.id !== processingMessage.id));
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        text: `Ошибка: ${error instanceof Error ? error.message : 'Неизвестная ошибка'}. Проверьте консоль для подробностей.`,
        sender: 'ai',
        timestamp: formatTimestamp(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsParsingText(false);
    }
  };

  // -------- attachments --------
  const handleAttachment = () => {
    setError(null);
    fileInputRef.current?.click();
  };
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (selectedChat !== 'financial-diary') {
      if (fileInputRef.current) fileInputRef.current.value = '';
      setError('Receipt uploads are only available inside the financial diary.');
      return;
    }

    setError(null);
    setIsUploadingReceipt(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-receipt`, {
        method: 'POST',
        body: formData,
      });

      let payload: any = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail = payload?.detail ?? payload?.message ?? 'Unable to process receipt.';
        throw new Error(typeof detail === 'string' ? detail : 'Unable to process receipt.');
      }

      if (!payload?.success) {
        const detail = payload?.message ?? 'Receipt was not processed.';
        throw new Error(typeof detail === 'string' ? detail : 'Receipt was not processed.');
      }

      const mappedEntries: DiaryTransaction[] = (payload.transactions ?? [])
        .map(mapRawTransaction)
        .filter(Boolean) as DiaryTransaction[];

      if (!mappedEntries.length) {
        throw new Error('Receipt did not return any transactions.');
      }

      setTransactions((prev) => {
        const combined = [...prev, ...mappedEntries];
        combined.sort((a, b) => parseTimestamp(a.date, a.time) - parseTimestamp(b.date, b.time));
        createCategoryColorLookup(combined.map((entry) => entry.category ?? '').filter(Boolean));
        return combined;
      });

      const successMessage: Message = {
        id: `receipt-${Date.now()}`,
        text: `Receipt uploaded: ${mappedEntries.length} entr${mappedEntries.length === 1 ? 'y' : 'ies'} added.`,
        sender: 'ai',
        timestamp: formatTimestamp(),
      };
      setMessages((prev) => [...prev, successMessage]);
      setTimeout(scrollToBottom, 200);
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : GENERAL_ERROR);
      }
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (isMountedRef.current) setIsUploadingReceipt(false);
    }
  };

  // -------- mic / STT --------
  const stopMediaStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  };

  const transcribeAudio = async (audioBlob: Blob) => {
    if (!audioBlob || audioBlob.size === 0) return;
    setIsTranscribing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', audioBlob, 'voice-message.webm');

    try {
      const res = await fetch(`${API_BASE_URL}/transcribe`, { method: 'POST', body: formData });
      if (!res.ok) {
        let detail = 'Zaman AI сейчас недоступен. Попробуйте позже.';
        try {
          const body = await res.json();
          if (typeof (body as any)?.detail === 'string') detail = (body as any).detail;
        } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      const text = (data?.text ?? '').trim();
      if (text) void submitMessage(text);
      else setError(TRANSCRIPTION_ERROR);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : GENERAL_ERROR);
    } finally {
      if (isMountedRef.current) setIsTranscribing(false);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      return;
    }
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setError(MICROPHONE_ERROR);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const mimeOptions = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      const supported = mimeOptions.find((m) => MediaRecorder.isTypeSupported(m));
      const recorder = new MediaRecorder(stream, supported ? { mimeType: supported } : undefined);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.addEventListener('dataavailable', (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      });
      recorder.addEventListener('stop', async () => {
        stopMediaStream();
        setIsRecording(false);
        const mimeType = recorder.mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        audioChunksRef.current = [];
        await transcribeAudio(blob);
      });

      recorder.start();
      setError(null);
      setIsRecording(true);
    } catch (err) {
      stopMediaStream();
      setError(MICROPHONE_ERROR);
    }
  };

  // -------- TTS replay --------
  const handlePlayLastResponse = async () => {
    const lastAssistantMessage = [...messages].reverse().find((m) => m.sender === 'ai');
    if (!lastAssistantMessage || isGeneratingAudio) return;

    setIsGeneratingAudio(true);
    setError(null);
    try {
      const audioBlob = await fetchAudioResponse(
        lastAssistantMessage.text,
        detectLanguage(lastAssistantMessage.text)
      );
      if (audioSourceRef.current) URL.revokeObjectURL(audioSourceRef.current);
      const audioUrl = URL.createObjectURL(audioBlob);
      audioSourceRef.current = audioUrl;

      const player = audioPlayerRef.current;
      if (!player) throw new Error('Аудиоэлемент ещё не готов.');
      player.src = audioUrl;
      await player.play();
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : AUDIO_ERROR);
    } finally {
      if (isMountedRef.current) setIsGeneratingAudio(false);
    }
  };

  // -------- UI status text --------
  const statusText = (() => {
    if (isUploadingReceipt) return 'Processing receipt...';
    if (isTranscribing) return 'Преобразуем запись в текст...';
    if (isProcessing) return 'Zaman AI готовит ответ...';
    if (isGeneratingAudio) return 'Готовим аудиоответ...';
    if (isParsingText) return 'Обрабатываю финансовый запрос...';
    return null;
  })();

  // ---------- Option screen (styled) ----------
  if (!selectedChat) {
    return (
      <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-lg dark:text-white">Zaman GPT</h1>
            <button className="p-2" aria-label="Меню">
              <Menu className="w-6 h-6 dark:text-white" />
            </button>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Выбери задачу — он справится за секунды
          </p>
        </div>

        {/* Chat Options */}
        <ScrollArea className="flex-1 px-4">
          <div className="py-4 space-y-3">
            {chatOptions.map((option) => (
              <button
                key={option.id}
                onClick={() => {
                  setSelectedChat(option.id);
                  setMessages([]);
                  setSelectedTransaction(null);
                }}
                className="w-full bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow text-left"
              >
                <div className="flex items-start gap-3">
                  <div className="w-14 h-14 rounded-full bg-[#EEFE6D] flex items-center justify-center flex-shrink-0">
                    {option.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium dark:text-white">{option.title}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {option.description}
                    </p>
                    {option.badge && (
                      <div className="inline-flex items-center gap-1 mt-2 px-2 py-1 bg-[#2D9A86] text-white rounded-full text-xs">
                        <Sparkles className="w-3 h-3" />
                        {option.badge}
                      </div>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>
    );
  }

  // ---------- Active chat screen ----------
  const currentChat = chatOptions.find((o) => o.id === selectedChat);
  const canReplayAudio = messages.some((m) => m.sender === 'ai');
  const disableComposer = isProcessing || isTranscribing || isParsingText || isUploadingReceipt;

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 px-4 py-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => setSelectedChat(null)} className="p-1" aria-label="Назад">
            <ChevronLeft className="w-6 h-6 dark:text-white" />
          </button>
          <div className="w-10 h-10 rounded-full bg-[#EEFE6D] flex items-center justify-center">
            {currentChat?.icon}
          </div>
          <div className="flex-1">
            <h1 className="dark:text-white">{currentChat?.title}</h1>
            <p className="text-xs text-gray-600 dark:text-gray-400">{currentChat?.description}</p>
          </div>

          {selectedChat === 'financial-diary' && transactions.length > 0 && (
            <button
              type="button"
              onClick={scrollToBottom}
              className="inline-flex items-center gap-2 rounded-full border border-gray-200 px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-100 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700"
            >
              <ChevronLeft className="h-4 w-4 rotate-90" />
              Вниз
            </button>
          )}

          <button
            type="button"
            onClick={handlePlayLastResponse}
            disabled={!canReplayAudio || isGeneratingAudio}
            className="inline-flex items-center gap-2 rounded-full border border-gray-200 px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700"
          >
            {isGeneratingAudio ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Воспроизводим...
              </>
            ) : (
              <>
                <Volume2 className="h-4 w-4" />
                Повтор
              </>
            )}
          </button>
        </div>
      </div>

      {/* Messages / Transactions */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea ref={scrollViewportRef} className="h-full px-4">
          <div className="py-4 space-y-4 pb-6">
            {selectedChat === 'financial-diary' ? (
              (transactions.length > 0 || messages.length > 0) ? (
                <>
                  {transactions.map((t) => {
                    const itemName = t.item && t.item.toLowerCase() !== 'unknown item' ? t.item : 'Без названия';
                    const categoryKey = t.category ?? undefined;
                    const iconLetter = (itemName?.trim()?.charAt(0) || 'T').toUpperCase();
                    const accentColor = getCategoryColor(categoryKey);
                    const isAccentLight = isHexColorLight(accentColor);
                    const iconTextClass = isAccentLight ? 'text-gray-900' : 'text-white';
                    const categoryTextClass = isAccentLight ? 'text-gray-900' : 'text-white';

                    const showActionRow = selectedTransaction === t.transactionId;
                    const hasCategory =
                      Boolean(t.category && t.category !== 'Other') || Boolean(t.categoryRu && t.categoryRu !== 'Прочее');
                    const hasItem = Boolean(t.item && t.item.toLowerCase() !== 'unknown item');
                    const categoryLabel = hasCategory ? t.categoryRu ?? t.category ?? '' : null;

                    return (
                      <div key={t.transactionId} className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm">
                        <div className="flex items-start gap-3">
                          <div
                            className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
                            style={{ backgroundColor: accentColor }}
                          >
                            <span className={`text-lg ${iconTextClass}`}>{iconLetter}</span>
                          </div>

                          <div className="flex-1">
                            <div className="flex justify-between items-start mb-1">
                              <div>
                                <div className="font-medium dark:text-white">{itemName || 'Покупка'}</div>
                                <div className="text-sm text-gray-600 dark:text-gray-400">
                                  Сумма: {formatCurrency(t.amount)}
                                </div>
                              </div>
                              <div className="text-right text-xs text-gray-400 dark:text-gray-500">
                                {t.date}
                                <br />
                                {t.time}
                              </div>
                            </div>
                            {hasCategory && categoryLabel ? (
                              <div
                                className={`mt-2 inline-block px-3 py-1 rounded-full text-xs ${categoryTextClass}`}
                                style={{ backgroundColor: accentColor }}
                              >
                                {categoryLabel}
                              </div>
                            ) : null}

                            {showActionRow && (
                              <div className="flex gap-2 mt-3 pt-3 border-t dark:border-gray-700">
                                <button
                                  className="flex-1 flex flex-col items-center gap-1 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                                  onClick={() => fileInputRef.current?.click()}
                                  aria-label="Прикрепить чек"
                                >
                                  <Camera className="w-5 h-5 text-[#2D9A86]" />
                                  <span className="text-xs text-gray-600 dark:text-gray-400">Прикрепить чек</span>
                                </button>
                                <button
                                  className="flex-1 flex flex-col items-center gap-1 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                                  aria-label="Выбрать тег"
                                >
                                  <Tag className="w-5 h-5 text-[#2D9A86]" />
                                  <span className="text-xs text-gray-600 dark:text-gray-400">Выбрать тег</span>
                                </button>
                                <button
                                  className="flex-1 flex flex-col items-center gap-1 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                                  onClick={() => {
                                    const hint = `Добавьте комментарий к транзакции ${t.transactionId}: `;
                                    setMessage((prev) => (prev ? prev : hint));
                                  }}
                                  aria-label="Комментарий"
                                >
                                  <MessageCircle className="w-5 h-5 text-[#2D9A86]" />
                                  <span className="text-xs text-gray-600 dark:text-gray-400">Комментарий</span>
                                </button>
                              </div>
                            )}

                            {!showActionRow && !hasCategory && (
                              <button
                                onClick={() => setSelectedTransaction(t.transactionId)}
                                className="w-full mt-3 pt-3 border-t dark:border-gray-700 text-center text-sm text-[#2D9A86] hover:text-[#268976]"
                              >
                                Добавить категорию
                              </button>
                            )}
                            {showActionRow && (
                              <button
                                onClick={() => setSelectedTransaction(null)}
                                className="w-full mt-3 pt-3 border-t dark:border-gray-700 text-center text-sm text-[#2D9A86] hover:text-[#268976]"
                              >
                                Скрыть действия
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {messages.map((msg) => {
                    const isSuccessMessage = msg.id.startsWith('success-');
                    const isErrorMessage = msg.id.startsWith('error-');
                    const isProcessingMessage = msg.id.startsWith('processing-');

                    if (isSuccessMessage) {
                      return (
                        <div key={msg.id} className="flex justify-start">
                          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800">
                            <p>{msg.text}</p>
                            <p className="text-xs mt-1 text-green-500 dark:text-green-400">{msg.timestamp}</p>
                          </div>
                        </div>
                      );
                    }

                    if (isProcessingMessage) {
                      return (
                        <div key={msg.id} className="flex justify-start">
                          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
                            <div className="flex items-center gap-2">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              <p>{msg.text}</p>
                            </div>
                            <p className="text-xs mt-1 text-blue-500 dark:text-blue-400">{msg.timestamp}</p>
                          </div>
                        </div>
                      );
                    }

                    if (isErrorMessage) {
                      return (
                        <div key={msg.id} className="flex justify-start">
                          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800">
                            <p>{msg.text}</p>
                            <p className="text-xs mt-1 text-red-500 dark:text-red-400">{msg.timestamp}</p>
                          </div>
                        </div>
                      );
                    }

                    return (
                      <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div
                          className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                            msg.sender === 'user'
                              ? 'bg-[#2D9A86] text-white'
                              : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white'
                          }`}
                        >
                          {renderMessageContent(msg)}
                          <p
                            className={`text-xs mt-1 ${
                              msg.sender === 'user' ? 'text-white/70' : 'text-gray-500 dark:text-gray-400'
                            }`}
                          >
                            {msg.timestamp}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </>
              ) : (
                <div className="text-center py-12">
                  <div className="w-20 h-20 rounded-full bg-[#EEFE6D] mx-auto mb-4 flex items-center justify-center">
                    <Wallet className="w-10 h-10" />
                  </div>
                  <p className="text-gray-600 dark:text-gray-400">Нет транзакций</p>
                  <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
                    Напишите что-то вроде «купил кофе за 500 тенге»
                  </p>
                </div>
              )
            ) : messages.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-20 h-20 rounded-full bg-[#EEFE6D] mx-auto mb-4 flex items-center justify-center">
                  {currentChat?.icon}
                </div>
                <p className="text-gray-600 dark:text-gray-400 mb-4">Начните диалог с {currentChat?.title}</p>
                {selectedChat === 'financial-diary' && (
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 max-w-md mx-auto">
                    <p className="text-sm text-blue-700 dark:text-blue-300 mb-2">
                      <strong>Как добавить расход:</strong>
                    </p>
                    <p className="text-xs text-blue-600 dark:text-blue-400">
                      Напишите в естественном формате:<br />
                      • «купил кока колу за 450 тенге»<br />
                      • «потратил 500 ₸ на кофе»<br />
                      • «заплатил за такси 1200 тг»
                    </p>
                  </div>
                )}
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      msg.sender === 'user'
                        ? 'bg-[#2D9A86] text-white'
                        : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white'
                    }`}
                  >
                    <p>{msg.text}</p>
                    <p
                      className={`text-xs mt-1 ${
                        msg.sender === 'user' ? 'text-white/70' : 'text-gray-500 dark:text-gray-400'
                      }`}
                    >
                      {msg.timestamp}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Composer */}
      <div className="bg-white dark:bg-gray-800 border-t dark:border-gray-700 p-4 flex-shrink-0">
        {error ? (
          <div className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200">
            {error}
          </div>
        ) : statusText ? (
          <div className="mb-3 flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{statusText}</span>
          </div>
        ) : null}

        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,audio/*,.pdf,.doc,.docx"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            onClick={handleAttachment}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            aria-label="Прикрепить файл"
            disabled={disableComposer && !messages.length}
          >
            <Paperclip className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>

          <button
            onClick={handleAttachment}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            aria-label="Открыть камеру / медиа"
            disabled={disableComposer && !messages.length}
          >
            <Camera className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>

          <button
            onClick={() => void toggleRecording()}
            className={`p-2 rounded-lg transition-colors ${
              isRecording ? 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400' : 'hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            aria-label={isRecording ? 'Остановить запись' : 'Начать запись'}
            disabled={!isRecording && disableComposer}
          >
            {isRecording ? <Loader2 className="w-5 h-5 animate-spin" /> : <Mic className="w-5 h-5 text-gray-600 dark:text-gray-400" />}
          </button>

          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              selectedChat === 'financial-diary'
                ? 'Например: «купил кока колу за 450 тенге сегодня» или «потратил 500 ₸ на кофе»'
                : 'Введите сообщение...'
            }
            className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
            disabled={disableComposer}
          />

          <Button
            onClick={handleSend}
            className="bg-[#2D9A86] hover:bg-[#268976]"
            disabled={disableComposer || !message.trim()}
            aria-label="Отправить сообщение"
          >
            {isProcessing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </Button>
        </div>
      </div>

      <audio ref={audioPlayerRef} className="hidden" />
    </div>
  );
}
