import { useState, useEffect } from 'react'
import { Button } from './components/ui/button'
import './App.css'
import Header from './ui components/Header'
import Sidebar from './ui components/Sidebar'
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar'
import { Textarea } from './components/ui/textarea'
import { Send, Moon, Sun, Menu } from 'lucide-react'
import axios from 'axios'

function SendIcon (props) {
  return (
    <svg
      {...props}
      xmlns='http://www.w3.org/2000/svg'
      width='24'
      height='24'
      viewBox='0 0 24 24'
      fill='none'
      stroke='currentColor'
      strokeWidth='2'
      strokeLinecap='round'
      strokeLinejoin='round'
    >
      <path d='m22 2-7 20-4-9-9-4Z' />
      <path d='M22 2 11 13' />
    </svg>
  )
}

function App () {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [file, setFile] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [topic, setTopic] = useState('')
  const toggleSidebar = () => setSidebarOpen(!sidebarOpen)

  useEffect(() => {
    // Fetch the topic when the app is loaded
    const fetchTopic = async () => {
      try {
        const response = await axios.get(
          'http://localhost:5000/api/prompt/topic'
        )
        setTopic(response.data.response)
      } catch (error) {
        console.error('Error fetching topic:', error)
      }
    }
    fetchTopic()
  }, [])

  const handleFileChange = event => {
    setFile(event.target.files[0])
  }

  const handleFileUpload = async () => {
    if (!file) {
      alert('Please select a file to upload.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(
        'http://localhost:5000/api/upload',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' }
        }
      )
      alert(response.data.response)
    } catch (error) {
      alert(
        `Error uploading file: ${error.response?.data?.error || error.message}`
      )
    }
  }

  const handleSendMessage = async () => {
    if (input.trim() !== '') {
      const newMessage = {
        id: messages.length + 1,
        text: input,
        sender: 'You',
        timestamp: new Date().toLocaleTimeString()
      }
      setMessages(prevMessages => [...prevMessages, newMessage])
      setInput('')

      try {
        const response = await fetchAIResponse(input)
        const aiMessage = {
          id: messages.length + 2,
          text: response,
          sender: 'AI',
          timestamp: new Date().toLocaleTimeString()
        }
        setMessages(prevMessages => [...prevMessages, aiMessage])
      } catch (error) {
        alert(`Error fetching AI response: ${error.message}`)
      }
    }
  }

  const fetchAIResponse = async prompt => {
    try {
      const response = await axios.post('http://localhost:5000/api/prompt', {
        prompt: prompt
      })
      if (response.data && response.data.response) {
        return response.data.response
      } else {
        throw new Error('Invalid response format from the server')
      }
    } catch (error) {
      if (error.response && error.response.data && error.response.data.error) {
        throw new Error(
          `Error fetching AI response: ${error.response.data.error}`
        )
      } else {
        throw new Error(`Error fetching AI response: ${error.message}`)
      }
    }
  }

  return (
    <div className='font-container flex flex-col min-h-screen'>
      <Header />
      <div className='flex-1 flex flex-col'>
        <section className='flex bg-zinc-900 flex-col flex-1 min-h-0'>
          <div className={`flex-1 overflow-auto p-4`}>
            <div className='lg:mx-24 mx-0 mt-8 mb-24 space-y-4'>
              {topic && (
                <div className='bg-zinc-800 text-white p-4 rounded-lg mb-4'>
                  <p>{topic}</p>
                </div>
              )}
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`flex items-start gap-3 ${
                    message.sender === 'You' ? 'justify-end' : ''
                  }`}
                >
                  {message.sender !== 'You' && (
                    <Avatar className='w-8 h-8'>
                      <AvatarImage src='/placeholder-user.jpg' />
                      <AvatarFallback>
                        {message.sender.charAt(0)}
                      </AvatarFallback>
                    </Avatar>
                  )}
                  <div
                    className={`rounded-2xl p-3 max-w-[70%] ${
                      message.sender === 'You'
                        ? 'bg-zinc-700 text-primary-foreground'
                        : 'bg-zinc-900 text-white'
                    }`}
                  >
                    <p className='text-base'>{message.text}</p>
                    <div
                      className={`text-base mt-1 ${
                        message.sender === 'You'
                          ? 'text-primary-foreground/80'
                          : 'text-muted-foreground'
                      }`}
                    >
                      {message.timestamp}
                    </div>
                  </div>
                  {message.sender === 'You' && (
                    <Avatar className='w-8 h-8'>
                      <AvatarImage src='/placeholder-user.jpg' />
                      <AvatarFallback>
                        {message.sender.charAt(0)}
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className='p-1 fixed bottom-0 w-full bg-zinc-800 shadow-inner'>
            <div className='flex my-4 mx-4 lg:mx-20 transition-all duration-300'>
              <textarea
                rows={1}
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder='Type your message here'
                className='flex-1 border-2 border-gray-400 bg-zinc-950 text-white p-4 rounded-l-full outline-none resize-none'
              />
              <button
                onClick={handleSendMessage}
                className='bg-gradient-to-r from-gray-100 to-white text-black rounded-r-full px-4 py-4 h-auto hover:opacity-90 transition duration-300 flex items-center self-center justify-center'
              >
                <Send size={30} />
              </button>
            </div>
            {/*<div className='flex items-center justify-between mx-4 lg:mx-20 mb-4'>
              <input
                type='file'
                onChange={handleFileChange}
                className='border-2 border-gray-400 bg-zinc-950 text-white p-2 rounded-md outline-none'
              />
              <button
                onClick={handleFileUpload}
                className='bg-gradient-to-r from-gray-100 to-white text-black rounded-md px-4 py-2 ml-2 hover:opacity-90 transition duration-300'
              >
                Upload File
              </button>
            </div>*/}
          </div>
        </section>
      </div>
    </div>
  )
}

export default App
