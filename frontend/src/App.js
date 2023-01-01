import "./App.css"

import { ChakraProvider } from "@chakra-ui/react"
import { Tabs, TabList, TabPanels, Tab, TabPanel } from "@chakra-ui/react"
import { Container } from "@chakra-ui/react"
import Instructor from "./components/Instructor"
import Student from "./components/Student"
import theme from "./theme"

function App() {
  return (
    <ChakraProvider theme={theme}>
      <div className="Dashboard">
        <Container maxW="800px">
          <Tabs size='lg' mt={4} variant="soft-rounded" colorScheme="green" isFitted>
            <TabList>
              <Tab>Instructor</Tab>
              <Tab>Student</Tab>
            </TabList>
            <TabPanels mt={10}  px={0}>
              <TabPanel className="Instructor">
                <Instructor></Instructor>
              </TabPanel>
              <TabPanel className="Student">
                <Student></Student>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Container>
      </div>
    </ChakraProvider>
  )
}

export default App
