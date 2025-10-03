# 🎯 Adaptive Minds Examples

This directory contains simple examples demonstrating how to use Adaptive Minds.

## 📋 Available Examples

### 1. 🐍 Python API Examples

**`basic_usage.py`** - Complete Python example showing:
- Server status checking
- Sending queries to different domain experts
- Handling responses and errors
- Clearing conversation history

```bash
# Run the basic examples
python examples/basic_usage.py
```

### 2. 🌐 cURL Examples

**`curl_examples.sh`** - Shell script with cURL commands:
- API endpoint examples
- JSON request formatting
- Response handling

```bash
# Make the script executable and run
chmod +x examples/curl_examples.sh
./examples/curl_examples.sh
```

### 3. 💬 Interactive Demo

**`interactive_demo.py`** - Command-line chat interface:
- Real-time conversation with Adaptive Minds
- Built-in help and commands
- Example queries and tips

```bash
# Start interactive chat
python examples/interactive_demo.py
```

## 🚀 Quick Start

1. **Start Adaptive Minds:**
   ```bash
   docker compose up
   ```

2. **Wait for initialization** (check logs for "🎉 ALL SYSTEMS READY!")

3. **Run any example:**
   ```bash
   # Python examples
   python examples/basic_usage.py
   python examples/interactive_demo.py
   
   # cURL examples  
   ./examples/curl_examples.sh
   ```

## 📊 Example Queries by Domain

| Domain | Example Query | Expected Expert |
|--------|---------------|----------------|
| **General** | "Hello, how are you?" | General |
| **Chemistry** | "What is the molecular formula of water?" | Chemistry |
| **Finance** | "How does compound interest work?" | Finance |
| **AI/Tech** | "Explain machine learning" | AI |
| **Medical** | "What are symptoms of flu?" | Medical |

## 🔧 API Endpoints Used

- **POST** `/chat` - Send queries
- **GET** `/status` - Check system status  
- **DELETE** `/chat/history` - Clear conversation history

## 🐛 Troubleshooting

**Connection errors:**
- Ensure Adaptive Minds is running: `docker compose up`
- Check server status: `curl http://localhost:8765/status`
- Wait for full initialization (~2-3 minutes on first run)

**Python requirements:**
```bash
pip install requests
```

## 💡 Next Steps

After trying these examples:

1. **Explore the Web UI:** http://localhost:8501
2. **Read the API docs:** Check the main README.md
3. **Build your own integration** using these examples as templates
4. **Try different query types** to see the intelligent routing in action

---

**Need help?** Check the main README.md or open an issue on GitHub!
