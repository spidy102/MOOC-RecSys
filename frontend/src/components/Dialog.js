import React from "react"
import axios from "axios"

import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  Button,
} from "@chakra-ui/react"
import { useDisclosure } from "@chakra-ui/react"

function Dialog({ isOpen, onOpen, onClose, handler, input }) {
  const cancelRef = React.useRef()

  return (
    <>
      {/* <Button colorScheme="red" onClick={onOpen}>
        Delete Customer
      </Button> */}

      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Ask Question
            </AlertDialogHeader>

            <AlertDialogBody>
              No similar questions were found in the database.. Would you like
              to post a new question?
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose}>
                Cancel
              </Button>
              <Button
                colorScheme="teal"
                onClick={() => {
                  onClose()
                  axios
                    .post(`http://127.0.0.1:8000/ask_quest`, {
                      quest: input,
                    })
                    .then((res) => {
                      console.log("success")
                    })
                }}
                ml={3}
              >
                Post
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  )
}

export default Dialog
