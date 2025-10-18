import { useEffect, useRef, useState } from 'react';
import { Camera, Mic, Send, Paperclip, Menu, Sparkles, FileText, Search, File, Wallet, ChevronLeft, MessageCircle, Loader2, Volume2 } from 'lucide-react';
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
    title: 'Financial Diary',
    description: 'Review recent spending, receipts, and quick insights.',
    icon: <Wallet className="w-6 h-6" />,
  },
  {
    id: 'zaman-ai',
    title: 'Zaman AI',
    description: 'Ask for personalised banking guidance powered by ZamanAI.',
    icon: <Sparkles className="w-6 h-6" />,
    badge: 'AI',
  },
  {
    id: 'text-work',
    title: 'Documents & Writing',
    description: 'Draft letters, summarise notes, or polish your pitch.',
    icon: <FileText className="w-6 h-6" />,
    badge: 'GPT-4o mini',
  },
  {
    id: 'ai-search',
    title: 'AI Search',
    description: 'Quick answers from trusted sources without leaving chat.',
    icon: <Search className="w-6 h-6" />,
  },
  {
    id: 'file-work',
    title: 'Files & Uploads',
    description: 'Send documents or images for instant discussion.',
    icon: <File className="w-6 h-6" />,
  },
];

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:8000';
const TIMESTAMP_LOCALE: Intl.LocalesArgument = 'ru-RU';
const formatTimestamp = () =>
  new Date().toLocaleTimeString(TIMESTAMP_LOCALE, { hour: '2-digit', minute: '2-digit' });
const detectLanguage = (text: string): string => (/[\u0400-\u04FF]/.test(text) ? 'ru' : 'en');
const GENERAL_ERROR = 'Unable to reach ZamanAI. Please try again later.';
const TRANSCRIPTION_ERROR = 'We could not understand your recording. Please try again.';
const MICROPHONE_ERROR = 'We could not access your microphone. Check permissions and try again.';
const AUDIO_ERROR = 'Unable to play the audio response. Please try again.';
const VOICE_UNAVAILABLE_ERROR = 'Voice transcription is not available with the current backend configuration.';

const extractChatReply = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const data = payload as Record<string, unknown>;
  const candidates = [data.response, data.reply, data.message];

  for (const candidate of candidates) {
    if (typeof candidate === 'string') {
      const trimmed = candidate.trim();
      if (trimmed) {
        return trimmed;
      }
    }
  }

  return null;
};

const decodeBase64Audio = (base64Audio: string): Uint8Array => {
  const binary = atob(base64Audio);
  const length = binary.length;
  const bytes = new Uint8Array(length);

  for (let index = 0; index < length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
};

const fetchAudioResponse = async (text: string, language: string): Promise<Blob> => {
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
        if (response.status === 404) {
          continue;
        }

        let detail: string | undefined;
        try {
          const errorBody = await response.json();
          detail = typeof errorBody?.detail === 'string' ? errorBody.detail : undefined;
        } catch {
          // Ignore parse errors
        }
        throw new Error(detail ?? 'ZamanAI is unavailable right now. Please try again later.');
      }

      const contentType = response.headers.get('content-type') ?? '';
      if (contentType.includes('application/json')) {
        const data = await response.json();
        const audioBase64 =
          (typeof data?.audio_base64 === 'string' && data.audio_base64) ||
          (typeof data?.audioBase64 === 'string' && data.audioBase64);

        if (!audioBase64) {
          const detail =
            (typeof data?.detail === 'string' && data.detail) ||
            (typeof data?.message === 'string' && data.message);
          throw new Error(detail ?? 'Unexpected audio payload from ZamanAI.');
        }

        const mimeType =
          (typeof data?.mime_type === 'string' && data.mime_type) ||
          (typeof data?.mimeType === 'string' && data.mimeType) ||
          'audio/mpeg';
        const audioBytes = decodeBase64Audio(audioBase64);
        return new Blob([audioBytes], { type: mimeType });
      }

      return await response.blob();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(GENERAL_ERROR);
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

      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop();
        } catch {
          // Ignore recorder shutdown issues
        }
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
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
    history.map((msg) => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.text,
    }));

  const submitMessage = async (content: string) => {
    const trimmed = content.trim();
    if (!trimmed || isProcessing) {
      return;
    }

    const historyPayload = mapHistory(messages);

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
          history: historyPayload,
        }),
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(VOICE_UNAVAILABLE_ERROR);
        }
        let detail = 'ZamanAI is unavailable right now. Please try again later.';
        try {
          const errorBody = await response.json();
          detail = errorBody.detail ?? detail;
        } catch {
          // Ignore body parse issues
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const aiText = extractChatReply(data);
      if (!aiText) {
        throw new Error('Unexpected response from ZamanAI.');
      }

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
    if (!trimmed) {
      return;
    }

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
      console.log('File selected:', file);
    }
  };

  const stopMediaStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  };

  const transcribeAudio = async (audioBlob: Blob) => {
    if (!audioBlob || audioBlob.size === 0) {
      return;
    }

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
        let detail = 'ZamanAI is unavailable right now. Please try again later.';
        try {
          const errorBody = await response.json();
          detail = errorBody.detail ?? detail;
        } catch {
          // Ignore parse issues
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
      if (isMountedRef.current) {
        setIsTranscribing(false);
      }
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

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
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
    const lastAssistantMessage = [...messages].reverse().find((msg) => msg.sender === 'ai');

    if (!lastAssistantMessage || isGeneratingAudio) {
      return;
    }

    setIsGeneratingAudio(true);
    setError(null);

    try {
      const audioBlob = await fetchAudioResponse(
        lastAssistantMessage.text,
        detectLanguage(lastAssistantMessage.text),
      );

      if (audioSourceRef.current) {
        URL.revokeObjectURL(audioSourceRef.current);
      }

      const audioUrl = URL.createObjectURL(audioBlob);
      audioSourceRef.current = audioUrl;

      const player = audioPlayerRef.current;
      if (!player) {
        throw new Error('Audio element is not ready.');
      }

      player.src = audioUrl;
      const playPromise = player.play();
      if (playPromise !== undefined) {
        await playPromise;
      }
    } catch (err) {
      console.error(err);
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : AUDIO_ERROR);
      }
    } finally {
      if (isMountedRef.current) {
        setIsGeneratingAudio(false);
      }
    }
  };

  const toggleTransactionActions = (transactionId: string) => {
    setExpandedTransaction((prev) => (prev === transactionId ? null : transactionId));
  };

  const statusText = (() => {
    if (error) {
      return error;
    }
    if (isTranscribing) {
      return 'Transcribing your message...';
    }
    if (isProcessing) {
      return 'ZamanAI is thinking...';
    }
    if (isGeneratingAudio) {
      return 'Preparing audio response...';
    }
    return null;
  })();

  if (!selectedChat) {
    return (
      <div className="flex h-full flex-col bg-gray-50 dark:bg-gray-900">
        <div className="border-b border-gray-200 bg-white px-4 py-4 dark:border-gray-700 dark:bg-gray-800">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">Zaman GPT</h1>
            <button className="rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-700" type="button" aria-label="Open chat menu">
              <Menu className="h-6 w-6 text-gray-700 dark:text-gray-200" />
            </button>
          </div>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Pick a conversation to get started, or open ZamanAI for tailored guidance.
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

  const activeChat = chatOptions.find((option) => option.id === selectedChat);
  const activeTitle = activeChat?.title ?? 'Conversation';
  const activeDescription = activeChat?.description ?? 'Chat with ZamanAI.';

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const renderTransactionList = () => {
    if (selectedChat !== 'financial-diary') {
      return null;
    }

    return (
      <section className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <header className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white">Recent Transactions</h2>
          <button
            type="button"
            className="inline-flex items-center text-xs font-medium text-blue-600 hover:underline dark:text-blue-300"
            onClick={() => setExpandedTransaction(null)}
          >
            Clear
          </button>
        </header>
        <div className="mt-3 space-y-3">
          {diaryTransactions.slice(0, 8).map((transaction) => (
            <div
              key={transaction.transactionId}
              className="rounded-lg border border-gray-100 bg-gray-50 p-3 dark:border-gray-700 dark:bg-gray-900"
            >
              <button
                type="button"
                className="flex w-full items-start justify-between text-left"
                onClick={() => toggleTransactionActions(transaction.transactionId)}
              >
                <div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {transaction.item ?? 'Transaction'}
                  </p>
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {transaction.date} · {transaction.time}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {formatCurrency(transaction.amount)}
                  </p>
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {transaction.category ?? 'Uncategorised'}
                  </p>
                </div>
              </button>
              {expandedTransaction === transaction.transactionId && (
                <div className="mt-3 space-y-2 text-xs text-gray-600 dark:text-gray-300">
                  <p>Customer: {transaction.customerId || '—'}</p>
                  <p>
                    Receipt: {transaction.hasReceipt ? 'Available in archive' : 'No receipt uploaded yet'}
                  </p>
                  {typeof transaction.quantity === 'number' && <p>Quantity: {transaction.quantity}</p>}
                  <div className="flex flex-wrap gap-2 pt-1 text-xs">
                    <Button variant="outline" className="h-7 px-3 text-xs">
                      View Details
                    </Button>
                    <Button variant="ghost" className="h-7 px-3 text-xs">
                      Flag
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    );
  };

  const canReplayAudio = messages.some((msg) => msg.sender === 'ai');
  const disableComposer = isProcessing || isTranscribing;

  return (
    <div className="flex h-full overflow-hidden bg-gray-50 dark:bg-gray-900">
      <aside className="hidden w-72 flex-col border-r border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900 md:flex">
        <div className="px-5 py-4">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white">Conversations</h2>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Jump into ZamanAI experiences tuned for everyday banking.
          </p>
        </div>
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-6">
          {chatOptions.map((option) => {
            const isActive = option.id === selectedChat;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => {
                  setSelectedChat(option.id);
                  setMessages([]);
                  setExpandedTransaction(null);
                }}
                className={`group relative flex w-full items-center gap-3 rounded-2xl border border-transparent px-3 py-3 text-left transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[#2D9A86] dark:focus-visible:ring-offset-gray-900 ${
                  isActive
                    ? 'bg-gradient-to-r from-[#2D9A86]/20 via-white to-white shadow-sm dark:from-[#2D9A86]/25 dark:via-gray-900 dark:to-gray-900'
                    : 'hover:border-[#2D9A86]/40 hover:bg-white/70 dark:hover:bg-gray-800/70'
                }`}
              >
                <span
                  className={`flex h-10 w-10 items-center justify-center rounded-xl text-white shadow-sm transition-all duration-200 ${
                    isActive
                      ? 'bg-gradient-to-br from-[#2D9A86] to-[#1F6F63]'
                      : 'bg-gray-100 text-[#2D9A86] shadow-none group-hover:text-[#1F6F63] dark:bg-gray-800 dark:text-[#EEFE6D]'
                  }`}
                >
                  {option.icon}
                </span>
                <div className="flex-1">
                  <p
                    className={`text-sm font-semibold leading-tight transition-colors ${
                      isActive ? 'text-[#1F6F63] dark:text-[#EEFE6D]' : 'text-gray-900 dark:text-white'
                    }`}
                  >
                    {option.title}
                  </p>
                  <p className="mt-1 text-xs text-gray-500 transition-colors duration-150 group-hover:text-gray-600 dark:text-gray-400 dark:group-hover:text-gray-300">
                    {option.description}
                  </p>
                </div>
                {option.badge && (
                  <span
                    className={`rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] transition-colors ${
                      isActive
                        ? 'border-[#2D9A86]/40 bg-[#EEFE6D]/90 text-[#1F6F63]'
                        : 'border-transparent bg-[#EEFE6D]/70 text-[#1F6F63] group-hover:border-[#2D9A86]/30'
                    } dark:border-[#2D9A86]/40 dark:bg-gray-800/80 dark:text-[#EEFE6D]`}
                  >
                    {option.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </aside>

      <main className="flex flex-1 flex-col">
        <header className="flex flex-col gap-4 border-b border-gray-200 bg-white px-4 py-4 dark:border-gray-700 dark:bg-gray-800 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setSelectedChat(null)}
              className="border border-transparent hover:border-gray-200 dark:hover:border-gray-700"
            >
              <ChevronLeft className="h-5 w-5" />
              <span className="sr-only">Back to chat options</span>
            </Button>
            <div>
              <h1 className="text-lg font-semibold text-gray-900 dark:text-white">{activeTitle}</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">{activeDescription}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => setMessages([])} disabled={messages.length === 0}>
              <MessageCircle className="h-4 w-4" />
              New Thread
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePlayLastResponse}
              disabled={!canReplayAudio || isGeneratingAudio}
            >
                {isGeneratingAudio ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating audio...
                  </>
                ) : (
                <>
                  <Volume2 className="h-4 w-4" />
                  Replay
                </>
              )}
            </Button>
          </div>
        </header>

        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 space-y-6 overflow-y-auto px-4 py-6 md:px-8">
            {renderTransactionList()}
            {messages.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-gray-300 bg-white/60 p-10 text-center text-gray-500 dark:border-gray-700 dark:bg-gray-900/40 dark:text-gray-400">
                <MessageCircle className="h-10 w-10" />
                <p className="mt-3 text-base font-medium">Start the conversation</p>
                <p className="mt-1 text-sm">
                  Ask ZamanAI to explain spending patterns, draft summaries, or search for insights.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg) => (
                  <div key={msg.id} className="flex flex-col gap-1">
                    <div
                      className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                        msg.sender === 'user'
                          ? 'self-end bg-blue-600 text-white'
                          : 'self-start bg-white text-gray-900 shadow-sm dark:bg-gray-800 dark:text-gray-100'
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
                    </div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {msg.sender === 'user' ? 'You' : 'ZamanAI'} · {msg.timestamp}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

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

            <div className="flex flex-wrap items-center gap-3">
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleFileChange}
                accept="audio/*,image/*,.pdf,.doc,.docx"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={handleAttachment}
                disabled={disableComposer && !messages.length}
              >
                <Paperclip className="h-5 w-5" />
                <span className="sr-only">Attach a file</span>
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={handleAttachment}
                disabled={disableComposer && !messages.length}
              >
                <Camera className="h-5 w-5" />
                <span className="sr-only">Capture media</span>
              </Button>
              <Button
                type="button"
                variant={isRecording ? 'destructive' : 'ghost'}
                size="icon"
                onClick={() => void toggleRecording()}
                disabled={!isRecording && disableComposer}
              >
                {isRecording ? <Loader2 className="h-5 w-5 animate-spin" /> : <Mic className="h-5 w-5" />}
                <span className="sr-only">{isRecording ? 'Stop recording' : 'Record voice message'}</span>
              </Button>
              <div className="flex-1 min-w-[180px]">
              <Input
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="Type your message..."
                disabled={disableComposer}
              />
              </div>
              <Button
                type="button"
                onClick={handleSend}
                disabled={disableComposer || !message.trim()}
                className="gap-2"
              >
                {isProcessing ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                Send
              </Button>
            </div>
          </div>
          <audio ref={audioPlayerRef} className="hidden" />
        </div>
      </main>
    </div>
  );
}
