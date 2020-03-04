---
layout: post
title: Facebook Analytics Academy Data Science Challenge
---
I was recently selected to compete in a Data Science challenge organised by Facebook at their London HQ. I didn't really have a lot of experience with data science tools, so I thought it would be a great platform to learn some cool new things. So I went into it with a very open mind and expected to totally bomb the whole thing (but hey, free trip to London!!)

They provided us with 3 separate Messenger datasets, consisting of sent messages data, received messages data and user profiles. They were huge csv files containing lots of information like times (sent, received and opened), country, platform used, OS used, etc. We were tasked to gain some insights from the data regarding health of the Messenger service, as well as propose hypotheses for what caused a major drop in sent messages on 25th December (surprise, it was Christmas).

For some strange reason, they decided to sort us into groups alphabetically? This meant that there were huge variations in experience between groups. I was placed into a group where everyone was like me and didn't have data science experience. There were two Imperial students, two Southampton students, and a lady from UC San Diego (she was in the UK on an exchange program).

I mostly worked on extracting insights by merging the datasets in various ways. I chose to use Pandas in Python to do most of the work. One of the cooler things I worked on was figuring out if there was a change in the proportion of bulk messages (i.e. same message sent to multiple people) sent out on Christmas. I figured that these messages would have the same message_id tagged to them, and extracting all the repetitions in that field would allow me to figure out the number of repeated messages. Turns out, there was actually a drop on Christmas, indicating that people preferred to send personalised one-to-one messages instead.

We must've done something right, because even though our group didn't have any experienced Statistics/Math students, we somehow ended up winning the challenge. It was really unexpected, but I was very pleased with the results. Got cool Facebook winners t-shirts!
