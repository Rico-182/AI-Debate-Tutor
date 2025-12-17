"""
Takes a youtube link, takes the transcription of it via a builtin GPT extension like YT Video transcripts.

Plugs the transcription into a prompt. e.g. 
+ Remove filler like uhms and transitions between speeches (usually I thank that speaker for that speech)
+ Take the proposition and opposition first speeches each. Return as a JSON
+ Guess what the motion debated is : and then have two educated guesses
Motion : {motion 1 guess} OR {motion 2 Guess}
Side: Government/Proposition OR Opposition. Depending on the context


Then have a python script take the JSON result, split it into two nice text blocks. 
Then we take each and plug it into another prmopt. Something like:

+ Clean this into a format that can be pasted into a txt
+ Sprinkle Metadata in each of the blocks for the corpus, so after 500-600 words repeat the metadata 
e.g. 
Sprinkle around:
Motion : {motion 1 guess} OR {motion 2 Guess}
Side: Government/Proposition OR Opposition. Depending on the context





"""