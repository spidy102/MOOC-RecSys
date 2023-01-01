import React from "react"
import { Input } from "@chakra-ui/react"
import { Button, Stack } from "@chakra-ui/react"
import { List, ListItem, ListIcon, OrderedList } from "@chakra-ui/react"
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Textarea,
  Box,
  Heading,
  useDisclosure,
} from "@chakra-ui/react"
import Dialog from "./Dialog"

import axios from "axios"

export default function Student() {
  const [input, setInput] = React.useState("")
  const [questions, setQuestions] = React.useState([])
  const [submitBtn, setSubmitBtn] = React.useState(false)
  const [answer, setAnswer] = React.useState("")
  const { isOpen, onOpen, onClose } = useDisclosure()

  const getQuestions = () => {
    axios
      .get(`http://127.0.0.1:8000/pred_student?quest=${input}`)
      .then((res) => {
        console.log(res.data)
        if (res.data[0] === 1) setQuestions(res.data[2])
        else if (res.data[0] === 0) onOpen()
        setSubmitBtn(false)
        //need to complete the part where new question to be added
      })
  }

  const handler = () => {
    axios
      .post(`http://127.0.0.1:8000/ask_quest`, {
        quest: input,
      })
      .then((res) => {
        console.log("success")
      })
  }
  const handleChange = (e) => setInput(e.target.value)
  return (
    <div>
      <Dialog
        isOpen={isOpen}
        onOpen={onOpen}
        onClose={onClose}
        handler={handler}
        input={input}
      />{" "}
      <Stack direction="row">
        <Input
          value={input}
          onChange={handleChange}
          variant="filled"
          size="lg"
          placeholder="Search for a question"
        />
        <Button
          isLoading={submitBtn}
          size="lg"
          colorScheme="teal"
          variant="solid"
          onClick={(e) => {
            setSubmitBtn(true)
            getQuestions()
          }}
        >
          Search
        </Button>
      </Stack>
      {/* <OrderedList>
        {questions.length === 0
          ? ""
          : questions.map((question) => {
              return <ListItem mb={4}>{question}</ListItem>
            })}
      </OrderedList> */}
      {questions.length > 0 ? (
        <Heading size="sm" textAlign="center" mt={4}>
          We found some similar questions
        </Heading>
      ) : (
        ""
      )}
      <Accordion allowToggle mt={5}>
        {questions.length === 0
          ? ""
          : questions.map((question) => {
              console.log("hello")
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
                    <Textarea value={question[1]} disabled />
                  </AccordionPanel>
                </AccordionItem>
              )
            })}
      </Accordion>
      <div
        style={{ marginTop: "100px", textAlign: "center", fontWeight: "bold" }}
      >
        {questions.length > 0 ? (
          <p>
            Didn't get an answer to your question?{" "}
            <a onClick={handler} style={{ textDecoration: "underline" }}>
              Ask
            </a>
          </p>
        ) : (
          ""
        )}
      </div>
    </div>
  )
}
