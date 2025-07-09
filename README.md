# Intelligent Systems Project - Voice AI Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20|%203.10-blue)
![Django](https://img.shields.io/badge/django-5.2-green)
![Kotlin](https://img.shields.io/badge/kotlin-1.9-purple)
![Next.js](https://img.shields.io/badge/next.js-15-black)
![TypeScript](https://img.shields.io/badge/typescript-5-blue)

> **Voice AI Assistant Project** - A comprehensive intelligent systems project featuring multiple voice recognition and AI response applications, from Kotlin desktop apps to modern web applications.

<div align="center">

![Voice AI Demo](docs/voice-ai-demo.gif)
<sub><i>Real-time voice detection, AI processing, and natural conversation flow</i></sub>
</div>

---

## 🚀 Project Overview

This project demonstrates the evolution of voice AI assistant technology through multiple implementations:

| Application | Technology Stack | Features |
|-------------|------------------|----------|
| **no11** (Desktop) | Kotlin + Swing + Java Sound API | Real-time voice detection, automatic recording, VAD |
| **no12** (Web App) | Next.js + TypeScript + Web Audio API | Modern UI, browser-based voice recognition, responsive design |
| **speechapi** | Django + REST API | Speech recognition backend service |
| **no06** (Legacy) | Django + YOLOv8 | Video analysis and pose detection |

---

## 🎯 Key Features

### Voice AI Assistant Core Features
- **🎤 Automatic Voice Detection**: Real-time audio level monitoring with configurable thresholds
- **🔇 Silence Detection**: Automatic recording stop after configurable silence duration
- **🤖 AI Integration**: Seamless integration with Ollama AI models (Gemma2, DeepSeek)
- **🗣️ Text-to-Speech**: Natural voice responses with language-specific synthesis
- **📊 Real-time Visualization**: Live audio level monitoring and status indicators
- **🔄 Continuous Conversation**: Automatic restart after AI response completion

### Technical Highlights
- **Cross-Platform**: Desktop (Kotlin) and Web (JavaScript) implementations
- **Modern Architecture**: RESTful APIs, reactive UI components, real-time processing
- **Voice Activity Detection (VAD)**: Advanced audio processing with customizable parameters
- **Responsive Design**: Modern UI with Tailwind CSS and shadcn/ui components

---

## 📁 Project Structure

```
ipro/
├── no11/                     # Kotlin Desktop Voice AI Assistant
│   ├── app/src/main/kotlin/  # Main application code
│   ├── build.gradle.kts      # Kotlin build configuration
│   └── README.md            # Kotlin app documentation
├── webapp/voice-ai-app/      # Next.js Web Voice AI Assistant
│   ├── src/app/             # Next.js app router pages
│   ├── src/components/ui/   # shadcn/ui components
│   ├── package.json         # Node.js dependencies
│   └── README.md           # Web app documentation
├── speechapi/               # Django Speech Recognition API
│   ├── views.py            # API endpoints
│   ├── models.py           # Database models
│   └── urls.py             # URL routing
├── no06/                   # Legacy Video Analysis (YOLOv8)
└── docs/                   # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+** (for Django API)
- **Node.js 18+** (for Web App)
- **Java 11+** (for Kotlin App)
- **Ollama** (for AI responses)

### 1. Clone Repository
```bash
git clone https://github.com/marutyan/Intelligent-Systems-Project.git
cd Intelligent-Systems-Project/ipro
```

### 2. Setup Ollama AI
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull required models
ollama pull gemma2
ollama pull deepseek-r1:7b
```

### 3. Setup Django Speech API
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start Django server
python manage.py runserver
```

### 4. Setup Web Application
```bash
# Navigate to web app
cd webapp/voice-ai-app

# Install dependencies
npm install

# Start development server
npm run dev
```

### 5. Setup Kotlin Application
```bash
# Navigate to Kotlin app
cd no11

# Build and run
./gradlew run
```

---

## 🎮 Usage Guide

### Web Application (Recommended)
1. **Start Services**:
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Start Django API
   python manage.py runserver
   
   # Terminal 3: Start Web App
   cd webapp/voice-ai-app && npm run dev
   ```

2. **Access Application**: Open `http://localhost:3000`

3. **Voice Interaction**:
   - Click the green microphone button to start voice monitoring
   - Speak naturally - recording starts automatically when voice is detected
   - AI responds with voice after 1.5 seconds of silence
   - Click red microphone button to stop monitoring

### Kotlin Desktop Application
1. **Start Services**: Ensure Ollama is running
2. **Launch App**: Run the Kotlin application
3. **Voice Interaction**: Same as web app, but with native desktop UI

---

## ⚙️ Configuration

### Voice Detection Parameters
```javascript
// Configurable in both applications
const SILENCE_THRESHOLD = 1000.0;      // Voice detection threshold
const SILENCE_DURATION = 1500;         // Silence timeout (ms)
const MIN_RECORDING_DURATION = 1500;   // Minimum recording time (ms)
```

### AI Model Configuration
```bash
# Switch between AI models
ollama pull gemma2          # Fast, efficient responses
ollama pull deepseek-r1:7b  # More detailed responses
ollama pull llama3.2        # Alternative model
```

### Environment Variables
```bash
# Django settings
export DJANGO_SECRET_KEY="your-secret-key"
export DEBUG=True

# Ollama settings
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="gemma2"
```

---

## 🔧 API Endpoints

### Speech Recognition API
```bash
# Upload audio for transcription
POST /api/upload/
Content-Type: multipart/form-data
Body: audio file (WAV format)

# Response
{
  "transcription": "recognized text"
}
```

### Ollama AI API
```bash
# Generate AI response
POST http://localhost:11434/api/generate
Content-Type: application/json
Body: {
  "model": "gemma2",
  "prompt": "user input",
  "stream": false
}
```

---

## 🧪 Development

### Running Tests
```bash
# Django tests
python manage.py test

# Web app tests
cd webapp/voice-ai-app
npm test

# Kotlin tests
cd no11
./gradlew test
```

### Code Quality
```bash
# Python formatting
black . && isort . && flake8 .

# JavaScript/TypeScript
npm run lint && npm run type-check

# Kotlin formatting
./gradlew ktlintFormat
```

### Building for Production
```bash
# Web application
cd webapp/voice-ai-app
npm run build
npm start

# Kotlin application
cd no11
./gradlew build
```

---

## 🐛 Troubleshooting

### Common Issues

#### Voice Recognition Not Working
- **Browser**: Use Chrome/Edge (Safari/Firefox have limitations)
- **Permissions**: Allow microphone access in browser
- **Protocol**: Use HTTPS or localhost only

#### AI Responses Not Generated
- **Ollama**: Check if `ollama serve` is running
- **Model**: Verify model is installed with `ollama list`
- **Network**: Check localhost:11434 connectivity

#### Audio Issues
- **Microphone**: Check system microphone settings
- **Volume**: Verify system volume and microphone levels
- **Format**: Ensure 16kHz sample rate support

#### Build Errors
```bash
# Clear Next.js cache
rm -rf .next

# Clear Kotlin build cache
./gradlew clean

# Reinstall dependencies
npm install
```

---

## 🚧 Roadmap

### Completed ✅
- [x] Kotlin desktop voice AI assistant
- [x] Web-based voice AI assistant
- [x] Real-time voice detection and processing
- [x] Ollama AI integration
- [x] Modern responsive UI
- [x] Cross-platform compatibility

### In Progress 🔄
- [ ] Mobile application (React Native)
- [ ] Voice training and personalization
- [ ] Multi-language support
- [ ] Cloud deployment setup

### Planned ⏩
- [ ] Voice emotion detection
- [ ] Conversation history persistence
- [ ] User authentication system
- [ ] Advanced voice commands
- [ ] Integration with external APIs

---

## 📊 Performance Metrics

| Metric | Kotlin App | Web App |
|--------|------------|---------|
| Voice Detection Latency | ~10ms | ~10ms |
| Speech Recognition | Django API | Web Speech API |
| AI Response Time | 2-5s | 2-5s |
| Memory Usage | ~50MB | ~30MB |
| CPU Usage | Low | Very Low |

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes**: Follow coding standards
4. **Add tests**: Ensure good test coverage
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**: Describe your changes

### Development Guidelines
- Follow existing code style and conventions
- Add comprehensive tests for new features
- Update documentation for API changes
- Use meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ollama** - Local AI model serving
- **OpenAI Whisper** - Speech recognition inspiration
- **Web Speech API** - Browser voice recognition
- **shadcn/ui** - Modern UI components
- **Next.js** - React framework
- **Kotlin** - JVM language for desktop apps

---

## 📞 Contact

- **Project Repository**: [GitHub](https://github.com/marutyan/Intelligent-Systems-Project)
- **Issues & Bug Reports**: [GitHub Issues](https://github.com/marutyan/Intelligent-Systems-Project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/marutyan/Intelligent-Systems-Project/discussions)

---

<div align="center">

**Built with ❤️ for Intelligent Systems Research**

*Demonstrating the evolution from desktop to web-based voice AI assistants*

</div>