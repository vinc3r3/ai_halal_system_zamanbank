import { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Mic,
  Send,
  Paperclip,
  Menu,
  Sparkles,
  FileText,
  Search,
  File,
  Wallet,
  ChevronLeft,
  Loader2,
  Volume2,
} from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { enrichedTransactions, formatCurrency } from '../data/financialData';

interface ChatOption {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  badge?: string;
}

const chatOptions: ChatOption[] = [
  {
    id: 'financial-diary',
    title: 'Финансовый дневник',
    description: 'Отслеживайте расходы и доходы',
    icon: <Wallet className="w-6 h-6" />,
  },
  {
    id: 'zaman-ai',
    title: 'Zaman AI',
    description:
      'Как взять кредит, копить и тратить, не нарушая законов Шариата',
    icon: <Sparkles className="w-6 h-6" />,
    badge: 'Халяль',
  },
  {
    id: 'text-work',
    title: 'Работа с текстом',
    description: 'Пишет за вас, подсказывает идеи',
    icon: <FileText className="w-6 h-6" />,
    badge: 'GPT-4o бесплатно',
  },
  {
    id: 'ai-search',
    title: 'ИИ-поисковик',
    description: 'Ищет ответы на любые вопросы',
    icon: <Search className="w-6 h-6" />,
  },
  {
    id: 'file-work',
    title: 'Работа с файлами',
    description: 'Ищет важное в файлах и объясняет простыми словами',
    icon: <File className="w-6 h-6" />,
  },
];

// --- ENV / helpers ---
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

const TIMESTAMP_LOCALE: Intl.LocalesArgument = 'ru-RU';
const formatTimestamp = () =>
  new Date().toLocaleTimeString(TIMESTAMP_LOCALE, {
    hour: '2-digit',
    minute: '2-digit',
  });

const detectLanguage = (text: string): 'ru' | 'en' =>
  /[\u0400-\u04FF]/.test(text) ? 'ru' : 'en';

// --- Fixed Russian strings (previously mojibake) ---
const GENERAL_ERROR = '?? ??????? ????????? ? Zaman AI. ????????? ??????? ?????.';
const TRANSCRIPTION_ERROR =
  'Не удалось распознать голосовое сообщение. Попробуйте ещё раз.';
const MICROPHONE_ERROR =
  'Нет доступа к микрофону. Проверьте разрешения и повторите попытку.';
const AUDIO_ERROR =
  'Не удалось воспроизвести аудиоответ. Попробуйте ещё раз.';
const VOICE_UNAVAILABLE_ERROR =
  'Голосовые функции недоступны для текущей конфигурации.';

const extractChatReply = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') return null;
  const data = payload as Record<string, unknown>;
  const candidates = [data.response, data.reply, data.message];

  for (const candidate of candidates) {
    if (typeof candidate === 'string') {
      const trimmed = candidate.trim();
      if (trimmed) return trimmed;
    }
  }
  return null;
};

const decodeBase64Audio = (base64Audio: string): Uint8Array => {
  const binary = atob(base64Audio);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
};

const fetchAudioResponse = async (text: string, language: string): Promise<Blob> => {
  // Try both endpoints for compatibility
  const endpoints = [`${API_BASE_URL}/tts`, `${API_BASE_URL}/speech`];
  const payload = JSON.stringify({ text, language });
  let lastError: Error | null = null;

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      });

      if (!response.ok) {
        if (response.status === 404) continue;
        let detail: string | undefined;
        try {
          const errorBody = await response.json();
          detail =
            typeof errorBody?.detail === 'string' ? errorBody.detail : undefined;
        } catch {
          // ignore
        }
        throw new Error(detail ?? 'Zaman AI сейчас недоступен. Попробуйте позже.');
      }

      const contentType = response.headers.get('content-type') ?? '';
      if (contentType.includes('application/json')) {
        const data = await response.json();

        // Accept `audio_base64` or `audioBase64` or `audio`
        const audioBase64 =
          (typeof (data as any)?.audio_base64 === 'string' && (data as any).audio_base64) ||
          (typeof (data as any)?.audioBase64 === 'string' && (data as any).audioBase64) ||
          (typeof (data as any)?.audio === 'string' && (data as any).audio);

        if (!audioBase64) {
          const detail =
            (typeof (data as any)?.detail === 'string' && (data as any).detail) ||
            (typeof (data as any)?.message === 'string' && (data as any).message);
          throw new Error(detail ?? 'Получен неожиданный аудиоответ от Zaman AI.');
        }

        const mimeType =
          (typeof (data as any)?.mime_type === 'string' && (data as any).mime_type) ||
          (typeof (data as any)?.mimeType === 'string' && (data as any).mimeType) ||
          'audio/mpeg';

        const audioBytes = decodeBase64Audio(audioBase64);
        return new Blob([audioBytes], { type: mimeType });
      }

      // If backend streams raw bytes
      return await response.blob();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(GENERAL_ERROR);
    }
  }

  throw lastError ?? new Error(GENERAL_ERROR);
};

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  attachment?: {
    type: 'image' | 'audio' | 'file';
    url: string;
  };
}

const diaryTransactions = enrichedTransactions;

export function ChatbotTab() {
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [expandedTransaction, setExpandedTransaction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioSourceRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== 'inactive'
      ) {
        try {
          mediaRecorderRef.current.stop();
        } catch {
          // ignore
        }
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

  useEffect(() => {
    setError(null);
  }, [selectedChat]);

  const mapHistory = (history: Message[]) =>
    history.map((m) => ({
      role: m.sender === 'user' ? 'user' : 'assistant',
      content: m.text,
    }));

  const submitMessage = async (content: string) => {
    const trimmed = content.trim();
    if (!trimmed || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: trimmed,
      sender: 'user',
      timestamp: formatTimestamp(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setError(null);
    setIsProcessing(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          history: mapHistory(messages),
        }),
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(VOICE_UNAVAILABLE_ERROR);
        }
        let detail = 'Zaman AI сейчас недоступен. Попробуйте позже.';
        try {
          const errorBody = await response.json();
          if (typeof errorBody?.detail === 'string') detail = errorBody.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const aiText = extractChatReply(data);
      if (!aiText) throw new Error('Неожиданный ответ от Zaman AI.');

      const aiMessage: Message = {
        id: `${Date.now()}-ai`,
        text: aiText,
        sender: 'ai',
        timestamp: formatTimestamp(),
      };

      if (isMountedRef.current) {
        setMessages((prev) => [...prev, aiMessage]);
      }
    } catch (err) {
      console.error(err);
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : GENERAL_ERROR);
      }
    } finally {
      if (isMountedRef.current) {
        setIsProcessing(false);
      }
    }
  };

  const handleSend = () => {
    const trimmed = message.trim();
    if (!trimmed) return;
    setMessage('');
    void submitMessage(trimmed);
  };

  const handleAttachment = () => {
    setError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // hook your upload/analysis pipeline here
      console.log('File selected:', file);
    }
  };

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
      const response = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detail = 'Zaman AI сейчас недоступен. Попробуйте позже.';
        try {
          const errorBody = await response.json();
          if (typeof errorBody?.detail === 'string') detail = errorBody.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const text = (data?.text ?? '').trim();

      if (text) {
        void submitMessage(text);
      } else {
        setError(TRANSCRIPTION_ERROR);
      }
    } catch (err) {
      console.error(err);
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : GENERAL_ERROR);
      }
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

      const mimeTypeOptions = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      const supportedMime =
        mimeTypeOptions.find((m) => MediaRecorder.isTypeSupported(m)) || '';

      const recorder = new MediaRecorder(stream, supportedMime ? { mimeType: supportedMime } : undefined);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) audioChunksRef.current.push(event.data);
      });

      recorder.addEventListener('stop', async () => {
        stopMediaStream();
        setIsRecording(false);

        const mimeType = recorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        audioChunksRef.current = [];

        await transcribeAudio(audioBlob);
      });

      recorder.start();
      setError(null);
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      stopMediaStream();
      setError(MICROPHONE_ERROR);
    }
  };

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

      if (audioSourceRef.current) {
        URL.revokeObjectURL(audioSourceRef.current);
      }

      const audioUrl = URL.createObjectURL(audioBlob);
      audioSourceRef.current = audioUrl;

      const player = audioPlayerRef.current;
      if (!player) throw new Error('Аудиоэлемент ещё не готов.');

      player.src = audioUrl;
      const playPromise = player.play();
      if (playPromise !== undefined) await playPromise;
    } catch (err) {
      console.error(err);
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : AUDIO_ERROR);
      }
    } finally {
      if (isMountedRef.current) setIsGeneratingAudio(false);
    }
  };

  const toggleTransactionActions = (transactionId: string) => {
    setExpandedTransaction((prev) => (prev === transactionId ? null : transactionId));
  };

  const statusText = (() => {
    if (isTranscribing) return 'Преобразуем запись в текст...';
    if (isProcessing) return 'Zaman AI готовит ответ...';
    if (isGeneratingAudio) return 'Готовим аудиоответ...';
    return null;
  })();

  if (!selectedChat) {
    return (
      <div className="flex h-full flex-col bg-gray-50 dark:bg-gray-900">
        <div className="border-b border-gray-200 bg-white px-4 py-4 dark:border-gray-700 dark:bg-gray-800">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">Zaman GPT</h1>
            <button
              className="rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-700"
              type="button"
              aria-label="Открыть меню чата"
            >
              <Menu className="h-6 w-6 text-gray-700 dark:text-gray-200" />
            </button>
          </div>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Выберите раздел, чтобы начать, или откройте Zaman AI для персональной помощи.
          </p>
        </div>

        <ScrollArea className="flex-1 px-4 pb-4">
          <div className="space-y-3 py-4">
            {chatOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => {
                  setSelectedChat(option.id);
                  setMessages([]);
                  setExpandedTransaction(null);
                }}
                className="group w-full rounded-2xl bg-white p-4 text-left shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#2D9A86] focus-visible:ring-offset-2 focus-visible:ring-offset-white dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-800/90 dark:focus-visible:ring-offset-gray-900"
              >
                <div className="flex items-start gap-4">
                  <span className="flex h-14 w-14 items-center justify-center rounded-full bg-[#EEFE6D] text-[#1F6F63] shadow-inner shadow-black/10">
                    {option.icon}
                  </span>
                  <div className="flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <h2 className="text-base font-semibold text-gray-900 transition-colors dark:text-white dark:group-hover:text-[#EEFE6D] group-hover:text-[#1F6F63]">
                        {option.title}
                      </h2>
                      {option.badge ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-[#2D9A86] px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
                          <Sparkles className="h-3 w-3" />
                          {option.badge}
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-2 text-sm text-gray-600 transition-colors dark:text-gray-300 group-hover:text-gray-700 dark:group-hover:text-gray-200">
                      {option.description}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>
    );
  }

  const activeChat = chatOptions.find((o) => o.id === selectedChat);
  const activeTitle = activeChat?.title ?? 'Диалог';
  const activeDescription = activeChat?.description ?? 'Общайтесь с Zaman AI.';

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const renderTransactionList = () => {
    if (selectedChat !== 'financial-diary') return null;

    const transactionsToShow = diaryTransactions.slice(0, 8);
    const accentClasses = [
      'bg-[#2D9A86]',
      'bg-[#FFB74D]',
      'bg-[#4C6EF5]',
      'bg-[#E57373]',
      'bg-[#7E57C2]',
    ];
    const pickAccent = (seed: string) => {
      if (!seed) return accentClasses[0];
      const code = seed.charCodeAt(0);
      const index = Number.isFinite(code) ? Math.abs(code) % accentClasses.length : 0;
      return accentClasses[index];
    };

    if (!transactionsToShow.length) {
      return (
        <div className="rounded-2xl bg-white p-6 text-center shadow-sm dark:bg-gray-800">
          <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-[#EEFE6D]">
            <Wallet className="h-10 w-10 text-[#1F6F63]" />
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Пока нет недавней активности.
          </p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {transactionsToShow.map((transaction) => {
          const label = transaction.item?.trim() || 'Покупка';
          const initial = label.charAt(0).toUpperCase() || 'T';
          const accentClass =
            pickAccent(label || transaction.category || transaction.transactionId);
          const isExpanded = expandedTransaction === transaction.transactionId;

          return (
            <div
              key={transaction.transactionId}
              className="rounded-2xl bg-white p-4 shadow-sm dark:bg-gray-800"
            >
              <div className="flex items-start gap-3">
                <div
                  className={`flex h-12 w-12 items-center justify-center rounded-full text-white ${accentClass}`}
                >
                  <span className="text-lg font-semibold">{initial}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {label}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Сумма: {formatCurrency(transaction.amount)}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Категория: {transaction.category ?? 'Без категории'}
                      </p>
                    </div>
                    <div className="text-right text-xs text-gray-500 dark:text-gray-400">
                      {transaction.date}
                      <br />
                      {transaction.time}
                    </div>
                  </div>
                  {isExpanded ? (
                    <div className="mt-3 space-y-2 text-xs text-gray-600 dark:text-gray-300">
                      <p>Клиент: {transaction.customerId || '—'}</p>
                      <p>Чек: {transaction.hasReceipt ? 'Есть' : 'Нет'}</p>
                      {typeof transaction.quantity === 'number' ? (
                        <p>Количество: {transaction.quantity}</p>
                      ) : null}
                      <div className="flex flex-wrap gap-2 pt-1">
                        <button
                          type="button"
                          className="rounded-full bg-[#2D9A86] px-3 py-1 text-xs font-semibold text-white transition hover:bg-[#268976]"
                        >
                          Отметить проверенным
                        </button>
                        <button
                          type="button"
                          className="rounded-full border border-gray-200 px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-100 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700"
                        >
                          Добавить заметку
                        </button>
                        {transaction.hasReceipt ? (
                          <button
                            type="button"
                            className="rounded-full border border-gray-200 px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-100 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700"
                          >
                            Открыть чек
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="mt-3 border-t border-gray-100 pt-3 text-right dark:border-gray-700">
                <button
                  type="button"
                  onClick={() => toggleTransactionActions(transaction.transactionId)}
                  className="text-sm font-medium text-[#2D9A86] transition hover:text-[#268976]"
                >
                  {isExpanded ? 'Скрыть быстрые действия' : 'Показать быстрые действия'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const canReplayAudio = messages.some((m) => m.sender === 'ai');
  const disableComposer = isProcessing || isTranscribing;

  return (
    <div className="flex h-full flex-col bg-gray-50 dark:bg-gray-900">
      <div className="border-b border-gray-200 bg-white px-4 py-4 dark:border-gray-700 dark:bg-gray-800">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setSelectedChat(null)}
            className="rounded-lg p-1 transition hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label="Назад к разделам"
          >
            <ChevronLeft className="h-6 w-6 text-gray-700 dark:text-gray-200" />
          </button>
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#EEFE6D] text-[#1F6F63]">
            {activeChat?.icon}
          </div>
          <div className="flex-1">
            <h1 className="text-base font-semibold text-gray-900 dark:text-white">
              {activeTitle}
            </h1>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {activeDescription}
            </p>
          </div>
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

      <ScrollArea className="flex-1 px-4">
        <div className="space-y-4 py-4">
          {selectedChat === 'financial-diary' ? (
            renderTransactionList()
          ) : messages.length === 0 ? (
            <div className="py-12 text-center">
              <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-[#EEFE6D]">
                {activeChat?.icon}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Начните диалог: {activeChat?.title}.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                    msg.sender === 'user'
                      ? 'bg-[#2D9A86] text-white'
                      : 'bg-white text-gray-900 shadow-sm dark:bg-gray-800 dark:text-gray-100'
                  }`}
                >
                  <p>{msg.text}</p>
                  {msg.attachment ? (
                    <div className="mt-2 text-xs opacity-80">
                      <span className="inline-flex items-center rounded bg-white/20 px-2 py-0.5">
                        {msg.attachment.type.toUpperCase()} attachment
                      </span>
                    </div>
                  ) : null}
                  <p
                    className={`mt-2 text-xs ${
                      msg.sender === 'user'
                        ? 'text-white/70'
                        : 'text-gray-500 dark:text-gray-400'
                    }`}
                  >
                    {msg.sender === 'user' ? 'Вы' : 'Zaman AI'} • {msg.timestamp}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      <div className="border-t border-gray-200 bg-white px-4 py-4 dark:border-gray-700 dark:bg-gray-800">
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
            accept="audio/*,image/*,.pdf,.doc,.docx"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={handleAttachment}
            className="rounded-lg p-2 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 dark:hover:bg-gray-700"
            disabled={disableComposer && !messages.length}
            aria-label="Прикрепить файл"
          >
            <Paperclip className="h-5 w-5 text-gray-600 dark:text-gray-300" />
          </button>
          <button
            type="button"
            onClick={handleAttachment}
            className="rounded-lg p-2 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-60 dark:hover:bg-gray-700"
            disabled={disableComposer && !messages.length}
            aria-label="Открыть камеру / медиа"
          >
            <Camera className="h-5 w-5 text-gray-600 dark:text-gray-300" />
          </button>
          <button
            type="button"
            onClick={() => void toggleRecording()}
            className={`rounded-lg p-2 transition ${
              isRecording
                ? 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-300'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            disabled={!isRecording && disableComposer}
            aria-label={isRecording ? 'Остановить запись' : 'Начать запись'}
          >
            {isRecording ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Mic className="h-5 w-5 text-gray-600 dark:text-gray-300" />
            )}
          </button>

          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleInputKeyDown}
            placeholder={
              selectedChat === 'financial-diary'
                ? 'Спросите о последних тратах...'
                : 'Введите сообщение...'
            }
            disabled={disableComposer}
            className="flex-1 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          />

          <Button
            type="button"
            onClick={handleSend}
            disabled={disableComposer || !message.trim()}
            className="bg-[#2D9A86] hover:bg-[#268976]"
            aria-label="Отправить сообщение"
          >
            {isProcessing ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>

      <audio ref={audioPlayerRef} className="hidden" />
    </div>
  );
}
