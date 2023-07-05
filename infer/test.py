from urllib.parse import quote
import datetime
import asyncio
import aiohttp
total = 0
success = 0
fail = 0
# Asynchronous function to send a request and process the response
# Asynchronous function to send a request and process the response
async def send_request(url):
    global total
    global success
    global fail
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            total += 1
            # Ensure the response is successful
            if response.status == 200:
                success += 1
                # Read the response content as bytes
                audio_data = await response.read()

                # Save the audio data to a file
                filename = url.split('/')[4] +".mp3"  # Extract filename from URL
                with open(filename, 'wb') as f:
                    f.write(audio_data)

                return filename
            else:
                fail += 1
                return  response.json()


# List of URLs to send requests to

path = quote("http://res.cloudinary.com/dklakm8v6/video/upload/v1685079673/Musicfiles/yt1s.com_-_The_Weeknd_Starboy_ft_Daft_Punk_Official_Video_vphg2v.mp3")
urls = [
    # "http://184.105.3.254/predict/pekora/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/drake/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/kendrick/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/kanye/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/trump/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/biden/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/dua_lipa/" + path + "?transpose=0" ,
    # "http://184.105.3.254/predict/mjack/" + path + "?transpose=0",
    "http://184.105.3.254/predict/rihanna/" + path + "?transpose=0",
    "http://184.105.3.254/predict/jworld/" + path + "?transpose=0"
    # Add more endpoint URLs here
]

# Asynchronous main function to send requests concurrently
async def main():
    # Create a list to store the pending coroutines
    coroutines = []

    # Create a session and schedule the requests
    async with aiohttp.ClientSession() as session:
        for url in urls:
            coroutines.append(send_request(url))

        # Gather the results of the coroutines
        responses = await asyncio.gather(*coroutines)

        # Process the responses
        for response in responses:
            # Process the response here
            print(response)

# Run the main function
start_time = datetime.datetime.now()
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
elapsed_time = datetime.datetime.now() - start_time
print("Total requests sent:", total)
print("Successful requests:", success)
print("Failed requests:", fail)
hours = elapsed_time.seconds // 3600
minutes = (elapsed_time.seconds % 3600) // 60
seconds = elapsed_time.seconds % 60
print("Elapsed time: {}:{}:{}".format(hours, minutes, seconds))