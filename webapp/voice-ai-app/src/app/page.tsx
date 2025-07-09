'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Progress } from '@/components/ui/progress';
import { Mic, MicOff, Volume2, VolumeX, User, Bot } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// Web Speech API の型定義
declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    webkitSpeechRecognition: any;
  }
}

export default function Home() {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<'ready' | 'listening' | 'processing' | 'speaking'>('ready');
  const [isMuted, setIsMuted] = useState(false);
  const [progress, setProgress] = useState(0);
  
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    // Web Speech API の初期化
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SpeechRecognition = (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'ja-JP';

      recognitionRef.current.onstart = () => {
        setStatus('listening');
        setIsListening(true);
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        if (event.results[event.results.length - 1].isFinal) {
          handleUserInput(transcript);
        }
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setStatus('ready');
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        if (status === 'listening') {
          setStatus('ready');
        }
      };
    }

    // Text-to-Speech の初期化
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  const handleUserInput = async (transcript: string) => {
    if (!transcript.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: transcript,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setStatus('processing');
    setIsProcessing(true);
    setProgress(0);

    try {
      // プログレスバーのアニメーション
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Ollama APIに送信
      const response = await fetch('/api/ollama/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gemma2',
          prompt: `${transcript}。日本語で、簡潔に答えてください。`,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const data = await response.json();
      let aiResponse = data.response || 'すみません、応答を生成できませんでした。';

      // レスポンスのクリーニング
      aiResponse = aiResponse.replace(/<think>[\s\S]*?<\/think>/g, '');
      aiResponse = aiResponse.replace(/[a-zA-Z\s]+/g, '').trim();

      clearInterval(progressInterval);
      setProgress(100);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: aiResponse,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Text-to-Speech で読み上げ
      if (!isMuted && synthRef.current) {
        setStatus('speaking');
        const utterance = new SpeechSynthesisUtterance(aiResponse);
        utterance.lang = 'ja-JP';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        utterance.onend = () => {
          setStatus('ready');
          setIsProcessing(false);
          setProgress(0);
        };
        
        synthRef.current.speak(utterance);
      } else {
        setStatus('ready');
        setIsProcessing(false);
        setProgress(0);
      }

    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'エラーが発生しました。もう一度お試しください。',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setStatus('ready');
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (synthRef.current) {
      synthRef.current.cancel();
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'listening': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'speaking': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'listening': return '音声を聞いています...';
      case 'processing': return 'AI が考えています...';
      case 'speaking': return '応答中...';
      default: return 'マイクボタンを押して話しかけてください';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* ヘッダー */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">Voice AI Assistant</h1>
          <p className="text-slate-600">音声でAIと自然に会話できるアシスタント</p>
        </div>

        {/* ステータス表示 */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}></div>
                <span className="text-sm font-medium text-slate-700">{getStatusText()}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant={status === 'ready' ? 'secondary' : 'default'}>
                  {status === 'ready' ? '待機中' : 
                   status === 'listening' ? '聞き取り中' :
                   status === 'processing' ? '処理中' : '応答中'}
                </Badge>
              </div>
            </div>
            {isProcessing && (
              <div className="mt-4">
                <Progress value={progress} className="w-full" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* 音声コントロール */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex justify-center space-x-4">
              <Button
                onClick={toggleListening}
                size="lg"
                className={`w-16 h-16 rounded-full ${
                  isListening 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-green-500 hover:bg-green-600'
                }`}
                disabled={isProcessing}
              >
                {isListening ? (
                  <MicOff className="w-6 h-6" />
                ) : (
                  <Mic className="w-6 h-6" />
                )}
              </Button>
              <Button
                onClick={toggleMute}
                size="lg"
                variant="outline"
                className="w-16 h-16 rounded-full"
              >
                {isMuted ? (
                  <VolumeX className="w-6 h-6" />
                ) : (
                  <Volume2 className="w-6 h-6" />
                )}
              </Button>
            </div>
            <div className="text-center mt-4 text-sm text-slate-600">
              {isListening ? '録音を停止するにはマイクボタンを押してください' : 
               '音声入力を開始するにはマイクボタンを押してください'}
            </div>
          </CardContent>
        </Card>

        {/* 会話履歴 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <span>会話履歴</span>
              <Badge variant="secondary">{messages.length} 件</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {messages.length === 0 ? (
                <div className="text-center text-slate-500 py-8">
                  まだ会話がありません。<br />
                  マイクボタンを押して話しかけてください。
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`flex items-start space-x-2 max-w-xs lg:max-w-md ${
                        message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                      }`}
                    >
                      <Avatar className="w-8 h-8">
                        <AvatarFallback>
                          {message.type === 'user' ? (
                            <User className="w-4 h-4" />
                          ) : (
                            <Bot className="w-4 h-4" />
                          )}
                        </AvatarFallback>
                      </Avatar>
                      <div
                        className={`rounded-lg p-3 ${
                          message.type === 'user'
                            ? 'bg-blue-500 text-white'
                            : 'bg-slate-100 text-slate-800'
                        }`}
                      >
                        <p className="text-sm">{message.content}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {message.timestamp.toLocaleTimeString('ja-JP')}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* フッター */}
        <div className="text-center mt-8 text-sm text-slate-500">
          <p>Powered by Ollama AI • Web Speech API</p>
        </div>
      </div>
    </div>
  );
}
