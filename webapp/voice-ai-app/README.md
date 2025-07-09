# Voice AI Assistant Web App

モダンなUI/UXを持つ音声AIアシスタントWebアプリケーションです。

## 🚀 機能

- **音声認識**: Web Speech APIを使用した日本語音声認識
- **AI応答**: Ollama APIを使用したAI応答生成
- **音声合成**: Web Speech APIを使用した日本語音声合成
- **リアルタイム会話**: 音声での自然な会話体験
- **モダンUI**: Next.js + Tailwind CSS + shadcn/uiによる美しいインターフェース
- **レスポンシブデザイン**: デスクトップ・モバイル対応

## 🛠️ 技術スタック

- **Frontend**: Next.js 15, TypeScript, Tailwind CSS, shadcn/ui
- **Icons**: Lucide React
- **API**: Web Speech API, Ollama API
- **音声認識**: webkitSpeechRecognition
- **音声合成**: speechSynthesis

## 📋 必要な環境

1. **Node.js** (v18以上)
2. **Ollama** (ローカルで実行中)
   - モデル: `gemma2`
   - ポート: `11434`

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
npm install
```

### 2. Ollamaの起動

```bash
# Ollamaサーバーを起動
ollama serve

# gemma2モデルの確認（別ターミナルで）
ollama list
```

### 3. 開発サーバーの起動

```bash
npm run dev
```

アプリケーションは `http://localhost:3000` で利用できます。

## 🎯 使用方法

1. **ブラウザでアクセス**: `http://localhost:3000`
2. **マイク許可**: ブラウザのマイク使用許可を有効にする
3. **音声入力**: 緑色のマイクボタンをクリックして話しかける
4. **AI応答**: AIが音声で応答します
5. **ミュート**: スピーカーボタンでAI音声をミュート可能

## 📱 UI/UX機能

### ステータス表示
- **待機中**: グレーの点滅
- **聞き取り中**: 緑色の点滅
- **処理中**: 黄色の点滅（プログレスバー付き）
- **応答中**: 青色の点滅

### 音声コントロール
- **マイクボタン**: 音声入力の開始/停止
- **スピーカーボタン**: AI音声のミュート/ミュート解除

### 会話履歴
- **ユーザー発言**: 青色の吹き出し（右側）
- **AI応答**: グレーの吹き出し（左側）
- **タイムスタンプ**: 各メッセージに時刻表示

## 🔧 カスタマイズ

### AIモデルの変更
`src/app/page.tsx`の以下の部分を編集：

```typescript
body: JSON.stringify({
  model: 'gemma2', // ここを変更
  prompt: `${transcript}。日本語で、簡潔に答えてください。`,
  stream: false
})
```

### 音声設定の調整
```typescript
utterance.lang = 'ja-JP';  // 言語設定
utterance.rate = 1.0;      // 話速
utterance.pitch = 1.0;     // 音程
```

## 🐛 トラブルシューティング

### 音声認識が動作しない
- ブラウザのマイク許可を確認
- HTTPSまたはlocalhostでアクセス
- Chrome/Edge推奨（Safari/Firefoxは制限あり）

### AI応答が返ってこない
- Ollamaサーバーが起動しているか確認
- `ollama list`でgemma2モデルがあるか確認
- ネットワーク接続を確認

### 音声合成が動作しない
- ブラウザの音声合成対応を確認
- ミュートボタンの状態を確認
- システム音量を確認

## 📁 プロジェクト構造

```
voice-ai-app/
├── src/
│   ├── app/
│   │   ├── page.tsx          # メインページ
│   │   ├── layout.tsx        # レイアウト
│   │   └── globals.css       # グローバルスタイル
│   ├── components/
│   │   └── ui/              # shadcn/uiコンポーネント
│   └── lib/
│       └── utils.ts         # ユーティリティ関数
├── public/                  # 静的ファイル
├── next.config.ts          # Next.js設定
└── package.json           # 依存関係
```

## 🚀 本番環境への展開

```bash
# ビルド
npm run build

# 本番サーバー起動
npm start
```

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

---

**Powered by**: Next.js • Tailwind CSS • shadcn/ui • Ollama AI • Web Speech API
