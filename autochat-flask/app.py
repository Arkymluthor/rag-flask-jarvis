import asyncio
from quart import Quart,render_template,jsonify,request
from dotenv.main import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationBufferMemory
from rag_process.response_process import conversational_llm_response_with_memory
from rag_process.utilis import response_handler
load_dotenv(override=True)



app = Quart(__name__)
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question", k=5)

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/data', methods=['POST'])
async def get_data():
    data = await request.get_json()
    text= data.get('data')
    user_query = text
    try:
        output = await conversational_llm_response_with_memory(user_query,memory)
        reply,urls = response_handler(output)
        return jsonify({"response":True,"message":reply,"urls":urls})
    except Exception as e:
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})

    
if __name__ == '__main__':
        app.run()



    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.run_until_complete(app.run())

