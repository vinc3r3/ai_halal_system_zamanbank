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
import { useTheme } from '../contexts/ThemeContext';
import { enrichedTransactions, formatCurrency, getCategoryColor } from '../data/financialData';

// ---------- Types ----------
interface ChatOption {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  badge?: string;
}
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
  attachment?: { type: 'image' | 'audio' | 'file'; url: string };
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
  const [selectedTransaction, setSelectedTransaction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioSourceRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);

  const { theme } = useTheme(); // (kept for your context API; not required below but harmless)

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
      console.log('Sending message to API:', { message: trimmed, history: mapHistory(messages) });
      console.log('API URL:', `${API_BASE_URL}/chat`);
      
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, history: mapHistory(messages) }),
      });

      console.log('API Response status:', res.status);
      console.log('API Response headers:', res.headers);

      if (!res.ok) {
        if (res.status === 404) throw new Error(VOICE_UNAVAILABLE_ERROR);
        let detail = 'Zaman AI сейчас недоступен. Попробуйте позже.';
        try {
          const body = await res.json();
          console.log('Error response body:', body);
          if (typeof body?.detail === 'string') detail = body.detail;
        } catch (parseErr) {
          console.log('Failed to parse error response:', parseErr);
        }
        throw new Error(detail);
      }

      const data = await res.json();
      console.log('API Response data:', data);
      const aiText = extractChatReply(data);
      if (!aiText) {
        console.log('No AI text extracted from response:', data);
        throw new Error('Неожиданный ответ от Zaman AI.');
      }
      const aiMessage: Message = { id: `${Date.now()}-ai`, text: aiText, sender: 'ai', timestamp: formatTimestamp() };
      if (isMountedRef.current) setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error in submitMessage:', err);
      
      // Provide a fallback response when API is not available
      if (err instanceof Error && (err.message.includes('fetch') || err.message.includes('Failed to fetch'))) {
        const fallbackMessage: Message = { 
          id: `${Date.now()}-ai-fallback`, 
          text: 'Извините, Zaman AI временно недоступен. Пожалуйста, попробуйте позже или обратитесь в службу поддержки.', 
          sender: 'ai', 
          timestamp: formatTimestamp() 
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
    
    // For financial diary, handle transaction addition locally
    if (selectedChat === 'financial-diary') {
      handleFinancialDiaryMessage(trimmed);
    } else {
      void submitMessage(trimmed);
    }
  };

  const handleFinancialDiaryMessage = (content: string) => {
    // Simple parsing for financial diary entries
    const amountMatch = content.match(/(\d+(?:\.\d{2})?)\s*₸?/);
    const amount = amountMatch ? parseFloat(amountMatch[1]) : 0;
    
    if (amount > 0) {
      const newTransaction = {
        id: `manual-${Date.now()}`,
        text: `Добавлен расход: ${content}`,
        sender: 'ai' as const,
        timestamp: formatTimestamp(),
      };
      setMessages((prev) => [...prev, newTransaction]);
    } else {
      void submitMessage(content);
    }
  };

  // -------- attachments --------
  const handleAttachment = () => {
    setError(null);
    fileInputRef.current?.click();
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) console.log('File selected:', file); // hook your upload pipeline here
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
          if (typeof body?.detail === 'string') detail = body.detail;
        } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      const text = (data?.text ?? '').trim();
      if (text) void submitMessage(text);
      else setError(TRANSCRIPTION_ERROR);
    } catch (err) {
      console.error(err);
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
      console.error(err);
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
      console.error(err);
      if (isMountedRef.current) setError(err instanceof Error ? err.message : AUDIO_ERROR);
    } finally {
      if (isMountedRef.current) setIsGeneratingAudio(false);
    }
  };

  // -------- UI status text --------
  const statusText = (() => {
    if (isTranscribing) return 'Преобразуем запись в текст...';
    if (isProcessing) return 'Zaman AI готовит ответ...';
    if (isGeneratingAudio) return 'Готовим аудиоответ...';
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

  // ---------- Active chat screen (styled + functional) ----------
  const currentChat = chatOptions.find((o) => o.id === selectedChat);
  const canReplayAudio = messages.some((m) => m.sender === 'ai');
  const disableComposer = isProcessing || isTranscribing;

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

          {/* Replay last AI message (TTS) */}
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
        <ScrollArea className="h-full px-4">
          <div className="py-4 space-y-4 pb-6">
            {selectedChat === 'financial-diary' ? (
              enrichedTransactions.length > 0 ? (
                enrichedTransactions.slice(0, 8).map((t) => {
                  const iconLetter = (t.item?.trim()?.charAt(0) || 'T').toUpperCase();
                  const accentColor = getCategoryColor(t.category);
                  const isAccentLight = isHexColorLight(accentColor);
                  const iconTextClass = isAccentLight ? 'text-gray-900' : 'text-white';
                  const categoryTextClass = isAccentLight ? 'text-gray-900' : 'text-white';

                  const showActionRow = selectedTransaction === t.transactionId;
                  const hasCategory = Boolean(t.category);

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
                              <div className="font-medium dark:text-white">{t.item || 'Покупка'}</div>
                              <div className="text-sm text-gray-600 dark:text-gray-400">
                                Сумма: {formatCurrency(t.amount)}
                              </div>
                              {typeof t.balance === 'number' ? (
                                <div className="text-xs text-gray-500 dark:text-gray-500">
                                  Доступно: {t.balance.toLocaleString()} ₸
                                </div>
                              ) : null}
                            </div>
                            <div className="text-right text-xs text-gray-400 dark:text-gray-500">
                              {t.date}
                              <br />
                              {t.time}
                            </div>
                          </div>
                          {hasCategory ? (
                            <div
                              className={`mt-2 inline-block px-3 py-1 rounded-full text-xs ${categoryTextClass}`}
                              style={{ backgroundColor: accentColor }}
                            >
                              {t.category}
                            </div>
                          ) : null}

                          {/* Quick actions row */}
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

                          {/* Toggle actions CTA */}
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
                })
              ) : (
                <div className="text-center py-12">
                  <div className="w-20 h-20 rounded-full bg-[#EEFE6D] mx-auto mb-4 flex items-center justify-center">
                    <Wallet className="w-10 h-10" />
                  </div>
                  <p className="text-gray-600 dark:text-gray-400">Нет транзакций</p>
                </div>
              )
            ) : messages.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-20 h-20 rounded-full bg-[#EEFE6D] mx-auto mb-4 flex items-center justify-center">
                  {currentChat?.icon}
                </div>
                <p className="text-gray-600 dark:text-gray-400">Начните диалог с {currentChat?.title}</p>
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
        {/* status + errors */}
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
            placeholder={selectedChat === 'financial-diary' ? 'Добавить расход или прикрепить чек...' : 'Введите сообщение...'}
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
