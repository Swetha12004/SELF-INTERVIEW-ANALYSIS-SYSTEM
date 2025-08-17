const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const path = require('path');

// Load environment variables
dotenv.config();

// Check for API key
const API_KEY = process.env.GEN_AI_API_KEY;
if (!API_KEY) {
    console.error('ERROR: GEN_AI_API_KEY not found in environment variables');
    process.exit(1);
}

// Initialize Express app
const app = express();

// Configure CORS and middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Initialize Google AI
const genAI = new GoogleGenerativeAI(API_KEY);

// Chat context and prompts
const SYSTEM_PROMPT = `You are an AI interview assistant. Your role is to:
1. Help candidates prepare for job interviews
2. Provide relevant interview questions and feedback
3. Give constructive advice
4. Share industry-specific insights

Please maintain a professional and supportive tone.`;

// Initialize chat model
const model = genAI.getGenerativeModel({ 
    model: "gemini-pro"
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Chat endpoint
app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        console.log('Received message:', message);

        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Generate response
        const result = await model.generateContent(message);
        const response = result.response.text();
        console.log('Generated response:', response);

        res.json({ response });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ 
            error: 'Failed to generate response',
            details: error.message 
        });
    }
});

// Serve the chat interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'chat.html'));
});

// Error handling
app.use((req, res) => {
    res.status(404).json({ error: 'Not found' });
});

app.use((err, req, res, next) => {
    console.error(err);
    res.status(500).json({ 
        error: 'Server error',
        message: err.message 
    });
});

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`API Key loaded: ${API_KEY ? 'Yes' : 'No'}`);
});