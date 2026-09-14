use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Grid {
    pub(crate) cells: [u8; 81],
}

impl Grid {
    pub fn empty() -> Self {
        Self { cells: [0; 81] }
    }

    pub fn from_line(line: &str) -> Result<Self, String> {
        if line.len() != 81 {
            return Err(format!("expected 81 characters, got {}", line.len()));
        }

        let mut cells = [0; 81];
        for (index, byte) in line.bytes().enumerate() {
            cells[index] = match byte {
                b'0' | b'.' => 0,
                b'1'..=b'9' => byte - b'0',
                _ => return Err(format!("invalid character at position {}", index)),
            };
        }
        Ok(Self { cells })
    }

    pub fn cells(&self) -> &[u8; 81] {
        &self.cells
    }

    pub fn to_line(&self) -> String {
        self.cells
            .iter()
            .map(|&cell| {
                if cell == 0 {
                    '0'
                } else {
                    char::from(b'0' + cell)
                }
            })
            .collect()
    }

    pub fn is_valid(&self) -> bool {
        let mut rows = [0u16; 9];
        let mut columns = [0u16; 9];
        let mut boxes = [0u16; 9];

        for index in 0..81 {
            let digit = self.cells[index];
            if digit == 0 {
                continue;
            }
            if !(1..=9).contains(&digit) {
                return false;
            }
            let row = index / 9;
            let column = index % 9;
            let box_index = (row / 3) * 3 + column / 3;
            let bit = 1u16 << digit;
            if rows[row] & bit != 0 || columns[column] & bit != 0 || boxes[box_index] & bit != 0 {
                return false;
            }
            rows[row] |= bit;
            columns[column] |= bit;
            boxes[box_index] |= bit;
        }
        true
    }
}

impl fmt::Debug for Grid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Grid")
            .field(&self.to_line())
            .finish()
    }
}
