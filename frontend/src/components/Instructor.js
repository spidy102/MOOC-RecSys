import React from "react"
import axios from "axios"

import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Alert,
  AlertIcon,
  CloseButton,
  Heading,
  Select
} from "@chakra-ui/react"

import { Textarea, Box, Button } from "@chakra-ui/react"

export default function Instructor() {
  const [questions, setQuestions] = React.useState([])
  const [input, setInput] = React.useState("")
  const [submitBtn, setSubmitBtn] = React.useState(false)
  const [alert, setAlert] = React.useState(false)
  const [unique_courses, setUnique_courses] = React.useState([])
  const [allGroups, setAllGroups] = React.useState({})
  React.useEffect(() => {
    axios
      .get("http://127.0.0.1:8000/pred_inst")
      .then((res) => {
        console.log(res.data)
        setAllGroups(res.data)
        // setQuestions(res.data.slice(0, 10))
        const all_courses = Object.keys(res.data)
        console.log(all_courses)
        let some = []
        for (let i=0;i<all_courses.length;i++) {
          const cur_course = all_courses[i].split("/")
          some.push(cur_course)
        }
        setUnique_courses(some)
      })
      .catch((e) => console.log(e))
    
  }, [])

  const startUpFunc = () => {
    axios
      .get("http://127.0.0.1:8000/pred_inst")
      .then((res) => {
        setQuestions(res.data.slice(0, 10))
      })
      .catch((e) => console.log(e))
    
  }

  return (
    <div>
      {alert ? (
        <Alert status="success" variant="subtle">
          <AlertIcon />
          Answer Posted successfully
          <CloseButton
            alignSelf="flex-start"
            position="relative"
            right={-1}
            top={-1}
            onClick={() => setAlert(false)}
          />
        </Alert>
      ) : (
        ""
      )}
      {/* <OrderedList>
        {questions.length === 0
          ? ""
          : questions.map((question) => {
              return <ListItem mb={4} fontSize='md'>{question}</ListItem>
            })}
      </OrderedList>
       */}
      {questions.length > 0 ? (
        <Heading mb={10} textAlign="center">
          Top Questions
        </Heading>
      ) : (
        ""
      )}
      <Select mb={5} placeholder="Select a course" variant='filled' size='lg' onChange={(e)=>{
        setQuestions(allGroups[e.target.value])
      }}>
        {unique_courses.length===0?"":unique_courses.map((course)=>{
          return (
            <option value={course.join("/")}>{course[0]+": "+course[1]+" - "+course[2]}</option>
          )
        })}
      </Select>
      <Accordion allowToggle>
        {questions.length === 0
          ? ""
          : questions.map((question) => {
              return (
                <AccordionItem>
                  <h2>
                    <AccordionButton>
                      <Box flex="1" textAlign="left">
                        {question[0]}
                      </Box>
                      <AccordionIcon />
                    </AccordionButton>
                  </h2>
                  <AccordionPanel pb={4}>
                    <Textarea
                      placeholder="Type in your answer"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                    />
                    <Button
                      isLoading={submitBtn}
                      colorScheme="teal"
                      mt={4}
                      size="sm"
                      onClick={(e) => {
                        setSubmitBtn(true)
                        axios
                          .post(
                            `http://127.0.0.1:8000/answer_quest/${question[1]}`,
                            {
                              id: question[1],
                              answer: input,
                            }
                          )
                          .then((res) => {
                            setSubmitBtn(false)
                            setAlert(true)
                            startUpFunc()
                          })
                      }}
                    >
                      Post Answer
                    </Button>
                  </AccordionPanel>
                </AccordionItem>
              )
            })}
      </Accordion>
    </div>
  )
}
