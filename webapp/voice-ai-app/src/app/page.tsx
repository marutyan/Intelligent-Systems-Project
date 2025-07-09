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
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<'ready' | 'listening' | 'recording' | 'processing' | 'speaking'>('ready');
  const [isMuted, setIsMuted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  
  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  
  // Voice detection refs
  const monitoringIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  
  // Text-to-Speech ref
  const synthRef = useRef<SpeechSynthesis | null>(null);
  
  // 音声検出の閾値設定（Kotlinアプリと同じ）
  const SILENCE_THRESHOLD = 1000.0;  // 音声検出の閾値
  const SILENCE_DURATION = 1500;     // 無音が続く時間(ms)
  const MIN_RECORDING_DURATION = 1500; // 最小録音時間(ms)

  useEffect(() => {
    // Text-to-Speech の初期化
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }

    return () => {
      stopVoiceDetection();
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  const initializeAudio = async () => {
    try {
      // マイクの許可を取得
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      streamRef.current = stream;
      
      // Web Audio API の初期化
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      microphoneRef.current.connect(analyserRef.current);
      
      // MediaRecorder の初期化
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];
        processAudioBlob(audioBlob);
      };
      
      return true;
    } catch (error) {
      console.error('Audio initialization error:', error);
      return false;
    }
  };

  const calculateVolume = () => {
    if (!analyserRef.current) return 0;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    
    return Math.sqrt(sum / bufferLength);
  };

  const startVoiceDetection = async () => {
    if (isMonitoring) return;
    
    const audioInitialized = await initializeAudio();
    if (!audioInitialized) {
      alert('マイクの初期化に失敗しました。ブラウザの設定を確認してください。');
      return;
    }
    
    setIsMonitoring(true);
    setStatus('listening');
    
    let isCurrentlyRecording = false;
    let silenceStart = 0;
    let recordingStart = 0;
    
    // 音声レベルの監視を開始
    monitoringIntervalRef.current = setInterval(() => {
      const volume = calculateVolume();
      setVolumeLevel(volume);
      
      const currentTime = Date.now();
      
      if (volume > SILENCE_THRESHOLD) {
        // 音声検出
        if (!isCurrentlyRecording && status !== 'processing' && status !== 'speaking') {
          console.log('Voice detected, starting recording');
          isCurrentlyRecording = true;
          recordingStart = currentTime;
          setStatus('recording');
          
          // 録音開始
          audioChunksRef.current = [];
          recordingStartTimeRef.current = currentTime;
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive') {
            mediaRecorderRef.current.start();
          }
        }
        silenceStart = currentTime;
      } else {
        // 無音
        if (isCurrentlyRecording) {
          if (currentTime - silenceStart > SILENCE_DURATION &&
              currentTime - recordingStart > MIN_RECORDING_DURATION) {
            // 録音終了
            console.log('Silence detected, stopping recording');
            isCurrentlyRecording = false;
            setStatus('processing');
            
            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
              mediaRecorderRef.current.stop();
            }
          }
        }
      }
    }, 10); // 10ms間隔で監視（Kotlinアプリと同じ）
  };

  const stopVoiceDetection = () => {
    setIsMonitoring(false);
    setStatus('ready');
    setVolumeLevel(0);
    
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
      monitoringIntervalRef.current = null;
    }
    
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    microphoneRef.current = null;
    analyserRef.current = null;
  };

  const processAudioBlob = async (audioBlob: Blob) => {
    try {
      // Web Speech API で音声認識
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'ja-JP';
      
      // Blob を audio element で再生して音声認識
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        console.log('Recognized:', transcript);
        handleUserInput(transcript);
        URL.revokeObjectURL(audioUrl);
      };
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setStatus('listening');
        URL.revokeObjectURL(audioUrl);
      };
      
      recognition.onend = () => {
        URL.revokeObjectURL(audioUrl);
      };
      
      // 音声認識を開始
      recognition.start();
      
    } catch (error) {
      console.error('Audio processing error:', error);
      setStatus('listening');
    }
  };

  const handleUserInput = async (transcript: string) => {
    if (!transcript.trim()) {
      setStatus('listening');
      return;
    }

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
          setStatus('listening');
          setIsProcessing(false);
          setProgress(0);
        };
        
        synthRef.current.speak(utterance);
      } else {
        setStatus('listening');
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
      setStatus('listening');
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const toggleMonitoring = () => {
    if (isMonitoring) {
      stopVoiceDetection();
    } else {
      startVoiceDetection();
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
      case 'listening': return 'bg-blue-500';
      case 'recording': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'speaking': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'listening': return '音声を待機しています...';
      case 'recording': return '録音中...';
      case 'processing': return 'AI が考えています...';
      case 'speaking': return '応答中...';
      default: return isMonitoring ? '音声を監視中' : 'マイクボタンを押して音声監視を開始してください';
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
                    status === 'listening' ? '待機中' :
                    status === 'recording' ? '録音中' :
                    status === 'processing' ? '処理中' : '応答中'}
                 </Badge>
               </div>
             </div>
             {isProcessing && (
               <div className="mt-4">
                 <Progress value={progress} className="w-full" />
               </div>
             )}
             {isMonitoring && (
               <div className="mt-4">
                 <div className="flex items-center space-x-2">
                   <span className="text-xs text-slate-500">音声レベル:</span>
                   <div className="flex-1 bg-slate-200 rounded-full h-2">
                     <div 
                       className="bg-green-500 h-2 rounded-full transition-all duration-100"
                       style={{ width: `${Math.min(volumeLevel / 50, 1) * 100}%` }}
                     ></div>
                   </div>
                   <span className="text-xs text-slate-500">{Math.round(volumeLevel)}</span>
                 </div>
               </div>
             )}
           </CardContent>
         </Card>

        {/* 音声コントロール */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex justify-center space-x-4">
              <Button
                onClick={toggleMonitoring}
                size="lg"
                className={`w-16 h-16 rounded-full ${
                  isMonitoring 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-green-500 hover:bg-green-600'
                }`}
                disabled={isProcessing}
              >
                {isMonitoring ? (
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
              {isMonitoring ? '音声認識を停止するにはマイクボタンを押してください' : 
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
