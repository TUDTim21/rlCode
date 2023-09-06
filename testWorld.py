import unittest

from my_sum import sum


class TestWorld(unittest.TestCase):
    def test_executeAction(self):
        
        action = Action("break", (2, 2, 1), 2)
        action = Action("move", (1, 0, 0), 1)
        action = Action("refuel", None, 0)

        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()