

def response_handler(gen_response):
    urls=[]
    
    if type(gen_response) is dict:
        reply = gen_response.get('answer',"No information found")

        if len(reply) == 0:
            reply = "I am unable to process any information right now. Please rephrase."

        if ("sources" in gen_response) and (len(gen_response["sources"].split(","))>0):
            try:
                urls = [src for src in gen_response["sources"].split(",")]
            except Exception as error:
                urls=[]
    else:
        reply = gen_response

    return reply, urls